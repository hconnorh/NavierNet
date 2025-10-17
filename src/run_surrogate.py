import os
import csv
import torch
import numpy as np

from typing import Tuple

from ml import SmallGraphModel

def load_trained_model(ckpt_path: str, device: torch.device | None = None,
                       in_features_default: int = 6, 
                       hidden_default: int = 128) -> SmallGraphModel:
    """
    Load a trained SmallGraphModel checkpoint saved by the training routine.
    Falls back to sensible defaults if metadata is missing.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hidden = int(ckpt.get("hidden_features", hidden_default))
    in_feats = int(ckpt.get("in_features", in_features_default))
    model = SmallGraphModel(in_features=in_feats, hidden_features=hidden, 
                            dropout_p=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    if device:
        model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict_next_step(model: SmallGraphModel, t_index: int, data: dict, 
                      device: torch.device) -> Tuple[np.ndarray, np.ndarray, 
                                                     np.ndarray]:
    """
    Predict the pressure and velocity (u, v) at the next time step for all 
    nodes in a graph, given a trained surrogate model and input data for the 
    current step.

    Args:
        model (SmallGraphModel): Trained neural network model for graph-based 
                                 surrogate prediction.
        t_index (int): Index of the current time step (0-based).
        data (dict): Dictionary containing the node and edge features, as 
                     returned by `load_graph_data`. Expected keys: "node_x", 
                     "node_y", "node_flags", "edge_index", "U", "V", "P".
        device (torch.device): Torch device to use for computation

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - p_next: Predicted pressure at the next step for each node
            - u_next: Predicted u-velocity at the next step for each node
            - v_next: Predicted v-velocity at the next step for each node
    """
    node_x = torch.from_numpy(data["node_x"]).to(device)
    node_y = torch.from_numpy(data["node_y"]).to(device)
    node_flags = torch.from_numpy(data["node_flags"]).to(device).float()
    edge_index = torch.from_numpy(data["edge_index"]).to(device)

    U = torch.from_numpy(data["U"]).to(device)
    V = torch.from_numpy(data["V"]).to(device)
    P = torch.from_numpy(data["P"]).to(device)

    u_t = U[t_index]
    v_t = V[t_index]
    p_t = P[t_index]

    static_feats = torch.stack([node_x, node_y, node_flags], dim=1)   # [N,3]
    x_node = torch.stack([u_t, v_t, p_t], dim=1)                      # [N,3]
    x_node = torch.cat([x_node, static_feats], dim=1)                 # [N,6]
    pred_delta = model(x_node, edge_index)                            # [N,3]
    du, dv, dp = pred_delta[:, 0], pred_delta[:, 1], pred_delta[:, 2]

    u_next = (u_t + du).cpu().numpy()
    v_next = (v_t + dv).cpu().numpy()
    p_next = (p_t + dp).cpu().numpy()
    return p_next, u_next, v_next


def save_prediction_csv(out_path: str, step_out: int, x: np.ndarray, 
                        y: np.ndarray, p: np.ndarray, 
                        u: np.ndarray, v: np.ndarray, 
                        node_ids: np.ndarray | None = None) -> None:
    """
    Save node-level predictions to CSV.

    Args:
        out_path (str): Output CSV file path.
        step_out (int): Simulation step for predictions.
        x (np.ndarray): x-coordinates of nodes.
        y (np.ndarray): y-coordinates of nodes.
        p (np.ndarray): Predicted pressure values.
        u (np.ndarray): Predicted u-velocity values.
        v (np.ndarray): Predicted v-velocity values.
        node_ids (np.ndarray | None): Optional stable node IDs, or None for 
                                      sequential IDs.
    """
    vn = np.hypot(u, v)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = len(x)
    if node_ids is None:
        node_ids = np.arange(N, dtype=np.int64)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","node_id","n_x","n_y","p","u","v","vn"])
        for nid, xi, yi, pi, ui, vi, vni in zip(node_ids, x, y, p, u, v, vn):
            w.writerow([step_out, int(nid), float(xi), float(yi), float(pi), 
                        float(ui), float(vi), float(vni)])
    print(f"Wrote predictions to {out_path}")
