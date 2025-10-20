import os
import json
import torch
import numpy as np
import pandas as pd

from typing import Dict, Tuple, List
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


# === GENERAL UTILS ===

def get_device() -> torch.device:
    """Returns the best available torch device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def calculate_residuals(df_sim, df_ml):
    """
    Calculate absolute percentage residuals between simulation (truth) and ML 
    predictions for each node and each step.

    This function compares the predicted values from surrogate model 
    (df_ml) with ground-truth simulation results (df_sim). It merges the two 
    DataFrames on ['step','node_id'], aligns nodal coordinates 
    (rounded to 12 decimals for uniqueness), and computes the absolute 
    percentage residuals for each field: pressure (p), velocity components 
    (u, v), and normal velocity (vn).

    Args:
        df_sim (pd.DataFrame): df containing simulation (ground truth) results.
                               Must contain columns ['step', 'node_id', 'n_x', 
                               'n_y', 'p', 'u', 'v', 'vn'] 
        df_ml  (pd.DataFrame): df containing ML predictions with the same 
                               structure as df_sim.

    Returns:
        pd.DataFrame: df where p, u, v, vn are the absolute pct residuals, i.e.
                      |prediction - truth| / |truth| for each field.
    """
    # Round to match extractor's 12-decimal uniqueness
    df_sim[['n_x','n_y']] = df_sim[['n_x','n_y']].round(12)
    df_ml[['n_x','n_y']]  = df_ml[['n_x','n_y']].round(12)

    # Merge on node_id
    pd.merge(
        df_sim, df_ml, on=['step','node_id'], how='inner', validate='one_to_one'
    )

    df_tmp = pd.merge(
        df_sim, df_ml, on=['step','node_id'], suffixes=('_true','_pred'), how='inner'
    )

    df_res = pd.DataFrame({
        'node_id': df_tmp['node_id'],
        'n_x': df_tmp['n_x_true'],
        'n_y': df_tmp['n_y_true'],
        'p': ((df_tmp['p_pred'] - df_tmp['p_true'])/df_tmp['p_true']).abs(),
        'u': ((df_tmp['u_pred'] - df_tmp['u_true'])/df_tmp['u_true']).abs(),
        'v': ((df_tmp['v_pred'] - df_tmp['v_true'])/df_tmp['v_true']).abs(),
        'vn': ((df_tmp['vn_pred'] - df_tmp['vn_true'])/df_tmp['vn_true']).abs(),
        'step': df_tmp['step']
    })

    return df_res



# === ML CLASSES ===

class GraphSAGELayer(torch.nn.Module):
    """
    Minimal GraphSAGE-like layer implemented with vanilla PyTorch.

    This layer performs graph-based message passing, enabling each node to 
    aggregate information from its local neighborhood. For each node, the layer 
    computes the mean of its neighbors' features (mean aggregation), 
    concatenates this mean vector with the node's own feature vector 
    (self features), and then applies a linear transformation to this 
    concatenated vector. 

    The transformed features are passed through a ReLU activation. Dropout is 
    then applied during training to randomly zero out certain elements which
    helps prevent overfitting and improves model generalisation.

    The core message passing leverages `edge_index`, a tensor encoding the 
    directed graph's connectivity (source and target node pairs). A 
    bidirectional or undirected graph is recommended for effective aggregation 
    of neighborhood information. The output dimension is defined by the 
    `out_features` argument. This GraphSAGE layer forms the fundamental 
    building block for deeper models that use stacked message passing layers.
    """

    def __init__(self, in_features: int, out_features: int, 
                       dropout_p: float = 0.0) -> None:
        """
        Initialise a GraphSAGE layer with mean aggregation and linear 
        transformation.

        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node.
            dropout_p (float): Dropout probability after activation.
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features * 2, out_features)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, node_features: torch.Tensor, 
                      edge_index: torch.Tensor) -> torch.Tensor:
        """
        Perform a single message-passing step of the GraphSAGE layers

        Args:
            node_features (torch.Tensor): Node features of shape [num_nodes, 
                                          in_features].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges], 
                                       where the first row contains source node 
                                       indices and the second row contains 
                                       destination node indices for each 
                                       directed edge.

        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, 
                          out_features] for each node in the graph.

        Notes:
            - number of nodees is set by the number of nodes in the input 
              layer, i.e. the number of nodes in the mesh.
            - the "edge_index" tensor essentially dictates how the nodes are
              connected within the graph, and thus how information "flows"
              through the network.
        
        TODO: should play around with edge_indexing to determine impact on 
              model performance.
        """
        num_nodes = node_features.shape[0]
        in_features = node_features.shape[1]

        # edge_index: shape [2, E] with (src, dst)
        src = edge_index[0]
        dst = edge_index[1]

        # Sum neighbor features into destination nodes
        neighbor_sum = torch.zeros((num_nodes, in_features), 
                        device=node_features.device, dtype=node_features.dtype)
        neighbor_sum.index_add_(0, dst, node_features[src])

        # Degree for mean aggregation
        ones = torch.ones((dst.shape[0],), device=node_features.device, 
                                           dtype=node_features.dtype)
        degree = torch.zeros((num_nodes,), device=node_features.device, 
                                           dtype=node_features.dtype)
        degree.index_add_(0, dst, ones)
        degree = degree.clamp(min=1.0).unsqueeze(-1)
        neighbor_mean = neighbor_sum / degree

        # Combine self and neighbor representations
        combined = torch.cat([node_features, neighbor_mean], dim=1)
        h = self.linear(combined)
        h = self.activation(h)
        h = self.dropout(h)

        return h

class SmallGraphModel(torch.nn.Module):
    """
    Two-layer GraphSAGE-style model that predicts (du, dv, dp) per node.

    This model encapsulates two stacked GraphSAGELayer instances followed by a 
    linear head. The model operates as follows:
        - Each GraphSAGELayer ingests node features and an edge_index (defining 
          the graph structure) and performs message-passing aggregation from 
          neighboring nodes, updating the node features.
        - The first layer takes the input node features and produces a "hidden"
          representation.
        - The second layer further propagates and aggregates information via 
          GraphSAGELayer, enabling nodes to access information from two-hop 
          neighbors.
        - The output of the last GraphSAGELayer is passed through a linear 
          layer ('head'), which reduces the dimensionality to 3, corresponding 
          to the predicted (du, dv, dp) changes for each node.
    
    The model thus allows efficient local graph-based information propagation 
    and node-wise prediction, making use of the aggregation logic defined 
    within GraphSAGELayer.

    TODO: Two layers are used to balance model expressiveness and computational 
    efficiency. Should experiment with more/less layers to see if performance
    can be improved without overfitting.
    """

    def __init__(self, in_features: int, hidden_features: int = 128, 
                       dropout_p: float = 0.0) -> None:
        super().__init__()
        self.layer1 = GraphSAGELayer(in_features, hidden_features, dropout_p)
        self.layer2 = GraphSAGELayer(hidden_features, hidden_features,
                                     dropout_p)
        self.head = torch.nn.Linear(hidden_features, 3)

    def forward(self, node_features: torch.Tensor, 
                      edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply two GraphSAGELayer message-passing steps followed by a prediction
        head.

        Args:
            node_features (torch.Tensor): Node features of shape [num_nodes, 
                                            in_features].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges], 
                                        where the first row contains source 
                                        node indices and the second row 
                                        contains destination node indices for 
                                        each directed edge.

        Returns:
            torch.Tensor: Predictions per node, [num_nodes, 3] (du, dv, dp).
        """
        h = self.layer1(node_features, edge_index)
        h = self.layer2(h, edge_index)
        out = self.head(h)
        return out  # shape: [N, 3] for (du, dv, dp)


# === UTILS ===

def _compute_graph_divergence(u: torch.Tensor, v: torch.Tensor, 
                              edge_index: torch.Tensor, 
                              edge_attr: torch.Tensor) -> torch.Tensor:
    """
    Computes an approximate divergence per node on a graph, based on edge 
    differences. In the context of (fluid) velocity fields, a divergence value 
    near zero indicates local incompressibility at each node, i.e., that the f
    ield preserves volume locally.

    For each directed edge from node i to node j, characterised by spatial 
    displacement (dx, dy) and edge length dist, the function evaluates a flux 
    contribution along the edge as:

        flux = ((u_j - u_i) * dx + (v_j - v_i) * dy) / (dist + eps)

    where u and v are scalar fields defined on nodes (such as velocity 
    components), and eps is a small constant to prevent division by zero.

    Each edge's flux is summed into its destination node. The aggregated sum is 
    then normalised by the destination node's in-degree, yielding a per-node 
    divergence estimate.

    Args:
        u (torch.Tensor): Node-wise scalar values of shape [N] (component u).
        v (torch.Tensor): Node-wise scalar values of shape [N] (component v).
        edge_index (torch.Tensor): Edge list of shape [2, E], with [0] as 
                                   source, [1] as destination.
        edge_attr (torch.Tensor): Edge attributes of shape [E, 3]; columns 
                                  are (dx, dy, dist).

    Returns:
        torch.Tensor: Divergence estimate per node, shape [N].
    """
    eps = 1e-8 # prevent div by zero
    num_nodes = u.shape[0]

    src = edge_index[0]
    dst = edge_index[1]

    dx = edge_attr[:, 0]
    dy = edge_attr[:, 1]
    dist = edge_attr[:, 2]

    du = u[src] - u[dst]
    dv = v[src] - v[dst]
    flux = (du * dx + dv * dy) / (dist + eps)

    # Sum fluxes into destination nodes
    flux_sum = torch.zeros((num_nodes,), device=u.device, dtype=u.dtype)
    flux_sum.index_add_(0, dst, flux)

    # Average by node degree
    ones = torch.ones((dst.shape[0],), device=u.device, dtype=u.dtype)
    degree = torch.zeros((num_nodes,), device=u.device, dtype=u.dtype)
    degree.index_add_(0, dst, ones)
    degree = degree.clamp(min=1.0)
    div = flux_sum / degree

    return div


def load_graph_data(sim_name: str, case_name: str) -> Dict[str, np.ndarray]:
    """
    Load graph data from the training data directory.
    """
    base_dir = ROOT / "sims" / sim_name / "training_data" / "graph" / case_name
    graph_npz = base_dir / "graph.npz"
    ts_npz = base_dir / "timeseries.npz"
    meta_json = base_dir / "meta.json"

    if not graph_npz.exists() or not ts_npz.exists():
        raise SystemExit(f"Missing graph artifacts under {base_dir}")

    graph = np.load(graph_npz)
    ts = np.load(ts_npz)

    with open(meta_json, "r") as f:
        meta = json.load(f)

    data = {
        "node_x": graph["node_x"].astype(np.float32),
        "node_y": graph["node_y"].astype(np.float32),
        "node_flags": graph["node_flags"].astype(np.int8),
        "edge_index": graph["edge_index"].astype(np.int64),
        "edge_attr": graph["edge_attr"].astype(np.float32),
        "steps": ts["steps"].astype(np.int32),
        "U": ts["U"].astype(np.float32),
        "V": ts["V"].astype(np.float32),
        "P": ts["P"].astype(np.float32),
        "dU": ts["dU"].astype(np.float32),
        "dV": ts["dV"].astype(np.float32),
        "dP": ts["dP"].astype(np.float32),
        "meta": meta,
    }

    return data


# === GNN TRAINGING ===

def train_model(sim_name: str, case_name: str, epochs: int = 5, 
                lr: float = 1e-3, hidden_features: int = 128, 
                dropout_p: float = 0.0, step_stride: int = 1, 
                lambda_div: float = 1e-3, lambda_bnd: float = 1e-2, 
                print_state: bool = False) -> List[Dict[str, float]]:
    """
    Trains a two-layer GraphSAGE model to predict changes in velocity and 
    pressure fields (du, dv, dp) at each node of a mesh for a given simulation 
    and case.

    The function uses node and edge features extracted from mesh/graph data to 
    learn per-node delta predictions. The loss consists of:
        - Mean squared error (MSE) between predicted and true deltas,
        - A divergence penalty term (weighted by `lambda_div`) which encourages 
          the physical consistency of the predicted velocity field,
        - An additional boundary node penalty (weighted by `lambda_bnd`)
          for nodes on the boundary, enforcing physically realistic changes.

    Training and validation timesteps are split chronologically. The function 
    supports configuration of model size, dropout, optimizer learning rate, and 
    regularisation strengths.

    Args:
        sim_name (str): Simulation dataset name.
        case_name (str): Specific case name within the simulation.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for Adam optimizer.
        hidden_features (int): Hidden layer dimensionality of the model.
        dropout_p (float): Dropout probability.
        step_stride (int): Timestep stride (for downsampling).
        lambda_div (float): Weight of divergence loss term.
        lambda_bnd (float): Weight of boundary loss term.
        print_state (bool): If True, prints progress during training.

    Returns:
        List[Dict[str, float]]: Aggregated metrics for each epoch, including:
            - train_loss: Average training loss per epoch.
            - val_loss: Average validation loss per epoch.
            - val_rmse_du, val_rmse_dv, val_rmse_dp: RMSE metrics on delta 
                                                     predictions.
            - val_div: Average divergence penalty on validation set.
            - val_bnd: Average boundary node penalty on validation set.

    Notes:
        - The model trains with implicit residual addition (i.e., the full 
          state is "next = current + predicted_delta").
    """
    np_data = load_graph_data(sim_name, case_name)

    node_x = torch.from_numpy(np_data["node_x"])  # [N]
    node_y = torch.from_numpy(np_data["node_y"])  # [N]
    node_flags = torch.from_numpy(np_data["node_flags"]).long()  # [N]
    edge_index = torch.from_numpy(np_data["edge_index"])  # [2, E]
    edge_attr = torch.from_numpy(np_data["edge_attr"])  # [E, 3]

    U = torch.from_numpy(np_data["U"])  # [T, N]
    V = torch.from_numpy(np_data["V"])  # [T, N]
    P = torch.from_numpy(np_data["P"])  # [T, N]
    dU = torch.from_numpy(np_data["dU"])  # [T-1, N]
    dV = torch.from_numpy(np_data["dV"])  # [T-1, N]
    dP = torch.from_numpy(np_data["dP"])  # [T-1, N]

    Tm1 = dU.shape[0]
    num_nodes = node_x.shape[0]

    # Build static node features once
    static_feats = torch.stack([
        node_x, node_y, node_flags.to(node_x.dtype)
    ], dim=1)  # [N, 3]

    # Split timesteps into train/val
    all_steps = torch.arange(0, Tm1, dtype=torch.long)
    all_steps = all_steps[::max(1, step_stride)]
    num_val = max(1, int(0.1 * all_steps.shape[0]))
    train_steps = all_steps[:-num_val] if all_steps.shape[0] > 1 else all_steps
    val_steps = all_steps[-num_val:] if all_steps.shape[0] > 1 else all_steps

    device = get_device()
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    static_feats = static_feats.to(device)
    node_flags = node_flags.to(device)

    model = SmallGraphModel(in_features=6, hidden_features=hidden_features, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()

    def timestep_forward(t_index: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Current state
        u_t = U[t_index].to(device)
        v_t = V[t_index].to(device)
        p_t = P[t_index].to(device)

        # Targets (deltas)
        du_true = dU[t_index].to(device)
        dv_true = dV[t_index].to(device)
        dp_true = dP[t_index].to(device)

        # Build node features: [u, v, p, x, y, flag]
        x_node = torch.stack([u_t, v_t, p_t], dim=1)
        x_node = torch.cat([x_node, static_feats], dim=1)  # [N, 6]

        # Predict deltas
        pred_delta = model(x_node, edge_index)
        du_pred = pred_delta[:, 0]
        dv_pred = pred_delta[:, 1]
        dp_pred = pred_delta[:, 2]

        # MSE on deltas
        loss_mse = mse_loss(du_pred, du_true) + mse_loss(dv_pred, dv_true) + mse_loss(dp_pred, dp_true)

        # Predicted next state
        u_next_pred = u_t + du_pred
        v_next_pred = v_t + dv_pred
        p_next_pred = p_t + dp_pred

        # Divergence penalty on predicted next velocity
        div = _compute_graph_divergence(u_next_pred, v_next_pred, edge_index, edge_attr)
        loss_div = torch.mean(div * div)

        # Boundary-weighted MSE at next state (encourage next state accuracy on boundaries)
        boundary_mask = (node_flags == 1).to(u_t.dtype)
        if boundary_mask.sum() > 0:
            u_next_true = u_t + du_true
            v_next_true = v_t + dv_true
            p_next_true = p_t + dp_true
            bw = boundary_mask
            # Mean over boundary nodes only
            loss_bnd = (
                torch.sum((u_next_pred - u_next_true) ** 2 * bw) +
                torch.sum((v_next_pred - v_next_true) ** 2 * bw) +
                torch.sum((p_next_pred - p_next_true) ** 2 * bw)
            ) / (3.0 * torch.clamp(bw.sum(), min=1.0))
        else:
            loss_bnd = torch.tensor(0.0, device=device, dtype=u_t.dtype)

        total_loss = loss_mse + lambda_div * loss_div + lambda_bnd * loss_bnd

        with torch.no_grad():
            rmse_du = torch.sqrt(torch.mean((du_pred - du_true) ** 2)).item()
            rmse_dv = torch.sqrt(torch.mean((dv_pred - dv_true) ** 2)).item()
            rmse_dp = torch.sqrt(torch.mean((dp_pred - dp_true) ** 2)).item()
            metrics = {
                "loss": float(total_loss.item()),
                "rmse_du": rmse_du,
                "rmse_dv": rmse_dv,
                "rmse_dp": rmse_dp,
                "div": float(loss_div.item()),
                "bnd": float(loss_bnd.item()),
            }
        return total_loss, metrics

    # Training
    res = []
    epoch_iter = range(1, epochs + 1)if print_state else tqdm(range(1, epochs + 1)) 
    for epoch in epoch_iter:
        model.train()
        train_losses: List[float] = []
        for t in train_steps.tolist():
            optimizer.zero_grad(set_to_none=True)
            total_loss, _ = timestep_forward(t)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(total_loss.item()))

        # Validation
        model.eval()
        with torch.no_grad():
            val_metrics_accum: Dict[str, float] = {"loss": 0.0, "rmse_du": 0.0, "rmse_dv": 0.0, "rmse_dp": 0.0, "div": 0.0, "bnd": 0.0}
            for t in val_steps.tolist():
                total_loss, metrics = timestep_forward(t)
                for k in val_metrics_accum:
                    val_metrics_accum[k] += float(metrics[k])
            for k in val_metrics_accum:
                val_metrics_accum[k] /= max(1, len(val_steps))

        if print_state:
            print(f"Epoch {epoch:03d} | train_loss={np.mean(train_losses):.6f} | val: "
            f"loss={val_metrics_accum['loss']:.6f} rmse_du={val_metrics_accum['rmse_du']:.6f} "
            f"rmse_dv={val_metrics_accum['rmse_dv']:.6f} rmse_dp={val_metrics_accum['rmse_dp']:.6f} "
            f"div={val_metrics_accum['div']:.6f} bnd={val_metrics_accum['bnd']:.6f}")

        # Aggregate per-epoch metrics
        epoch_metrics: Dict[str, float] = {
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(val_metrics_accum['loss']),
            "val_rmse_du": float(val_metrics_accum['rmse_du']),
            "val_rmse_dv": float(val_metrics_accum['rmse_dv']),
            "val_rmse_dp": float(val_metrics_accum['rmse_dp']),
            "val_div": float(val_metrics_accum['div']),
            "val_bnd": float(val_metrics_accum['bnd']),
        }

        res.append({
            "epoch": epoch,
            **epoch_metrics,
        })

    # Save model
    out_dir = ROOT / "sims" / sim_name / "training_data" / "graph" / case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model_graphsage.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_features": 6,
        "hidden_features": hidden_features,
    }, model_path)
    print(f"Saved model checkpoint to: {model_path}")

    return res


if __name__ == "__main__":
    # Case parameters
    SIM_NAME = "example-model"
    CASE_NAME = 'case0'

    # Train GNN model
    train_model(sim_name=SIM_NAME, case_name=CASE_NAME, epochs=5, lr=1e-3,
                hidden_features=128, dropout_p=0.0, step_stride=5,
                lambda_div=1e-3, lambda_bnd=1e-2)
