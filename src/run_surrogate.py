import os
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from typing import Tuple

from ml import SmallGraphModel, load_graph_data, get_device

ROOT = Path(__file__).resolve().parent.parent

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


def run_all_predictions(model: SmallGraphModel, data: dict,
                        device: torch.device | None = None,
                        steps: int | None = None,
                        start_step: int = 0,
                        out_csv: str | None = None,
                        overwrite: bool = True) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Predict the next state for multiple consecutive timesteps (teacher-forced)
    and write all predicted states to a CSV with the standard schema.

    Returns a list of (p_next, u_next, v_next) arrays for each predicted step.
    """
    
    device = get_device() if device is None else device

    # Timeseries boundaries
    U = data["U"]
    T = int(U.shape[0])
    if T < 2:
        return []
    start_step = int(max(0, min(start_step, T - 2)))  # must have a next step
    max_steps = (T - 1) - start_step
    num_steps = int(max_steps if steps is None else min(steps, max_steps))

    # Output CSV path
    if out_csv is None:
        meta = data.get("meta", {})
        sim_nm = meta.get("sim_name", "unknown-sim")
        case_nm = meta.get("case_name", "unknown-case")
        out_dir = ROOT / "sims" / sim_nm / "training_data" / "rollouts"
        out_csv_path = out_dir / f"{case_nm}-teacher_forced.csv"
    else:
        out_csv_path = Path(out_csv)

    # Prepare static geometry for writing
    xs = np.asarray(data["node_x"], dtype=float)
    ys = np.asarray(data["node_y"], dtype=float)

    # Step indices (prefer dataset-provided step numbers)
    steps_arr = data.get("steps", np.arange(T, dtype=int))

    # Init/overwrite CSV and append per predicted step
    _init_rollout_csv(out_csv_path, overwrite=overwrite)
    
    for t in tqdm(range(start_step, start_step + num_steps)):
        p_next, u_next, v_next = predict_next_step(model, t, data, device)
        step_out = int(steps_arr[t + 1])
        _append_rollout_step(out_csv_path, step_out, xs, ys, p_next, u_next, v_next)

    print(f"Predictions written to: {out_csv_path}")
    return out_csv_path


def _init_rollout_csv(out_csv_path: Path, overwrite: bool = True) -> None:
    """
    Create or reset a rollout CSV with the standard header.
    """
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_csv_path.exists():
        out_csv_path.unlink()
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","node_id","n_x","n_y","p","u","v","vn"])


def _append_rollout_step(out_csv_path: Path, step_out: int,
                         xs: np.ndarray, ys: np.ndarray,
                         p: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    """
    Append one timestep worth of node rows to an existing rollout CSV.
    """
    vn = np.hypot(u, v)
    node_ids = np.arange(xs.shape[0], dtype=np.int64)
    with open(out_csv_path, "a", newline="") as f:
        w = csv.writer(f)
        for nid, xi, yi, pi, ui, vi, vni in zip(node_ids, xs, ys, p, u, v, vn):
            w.writerow([int(step_out), int(nid), float(xi), float(yi), float(pi), float(ui), float(vi), float(vni)])

def bootstrap_simulation(sim_name: str, case_name: str | None = None,
                         steps: int = 200, start_step: int = 0,
                         out_csv: str | None = None, overwrite: bool = True) -> str:
    """
    Roll forward purely with the surrogate starting from an initial observed state.

    This performs an autoregressive rollout: it takes the state at `start_step`
    from the graph dataset, then repeatedly applies the model's predicted deltas
    to produce the next state, writing each predicted state's nodewise values to
    a CSV compatible with simulation outputs.

    Columns: step,node_id,n_x,n_y,p,u,v,vn
    """
    # Resolve case directory containing graph artifacts and checkpoint
    graph_root = ROOT / "sims" / sim_name / "training_data" / "graph"
    if case_name is None:
        candidates = []
        if graph_root.exists():
            for d in sorted(p.name for p in graph_root.iterdir() if p.is_dir()):
                if (graph_root / d / "model_graphsage.pt").exists():
                    candidates.append(d)
        if not candidates:
            # Fallback to first case name under pyfr_results if graph not present
            pr_root = ROOT / "sims" / sim_name / "pyfr_results"
            cases = [p.name for p in sorted(pr_root.iterdir()) if p.is_dir()] if pr_root.exists() else []
            if not cases:
                raise SystemExit(f"Could not infer case_name under {graph_root} or {pr_root}")
            case_name = cases[0]
        else:
            case_name = candidates[0]

    ckpt_path = graph_root / case_name / "model_graphsage.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Model checkpoint not found: {ckpt_path}")

    # Load data and model
    data = load_graph_data(sim_name, case_name)
    device = get_device()
    model = load_trained_model(str(ckpt_path), device=device)

    # Static graph tensors
    node_x = torch.from_numpy(data["node_x"]).to(device)
    node_y = torch.from_numpy(data["node_y"]).to(device)
    node_flags = torch.from_numpy(data["node_flags"]).to(device).float()
    edge_index = torch.from_numpy(data["edge_index"]).to(device)
    static_feats = torch.stack([node_x, node_y, node_flags], dim=1)  # [N,3]

    # Initial state
    U = data["U"]; V = data["V"]; P = data["P"]
    if U.shape[0] == 0:
        raise SystemExit("Empty timeseries arrays in graph dataset")
    start_step = int(max(0, min(start_step, U.shape[0] - 1)))
    u_cur = torch.from_numpy(U[start_step]).to(device)
    v_cur = torch.from_numpy(V[start_step]).to(device)
    p_cur = torch.from_numpy(P[start_step]).to(device)

    # Output path
    if out_csv is None:
        out_dir = ROOT / "sims" / sim_name / "training_data" / "rollouts"
        out_csv_path = out_dir / f"{case_name}-bootstrap.csv"
    else:
        out_csv_path = Path(out_csv)


    # Step base (match simulation-style step indexing when available)
    base_step = int(data.get("steps", np.arange(U.shape[0]))[start_step])

    # Init CSV (overwrite on reruns), then append per predicted step
    N = node_x.shape[0]
    xs = node_x.detach().cpu().numpy()
    ys = node_y.detach().cpu().numpy()
    _init_rollout_csv(out_csv_path, overwrite=overwrite)

    for i in tqdm(range(1, int(steps) + 1)):
        # Build node features and predict deltas
        x_node = torch.stack([u_cur, v_cur, p_cur], dim=1)
        x_node = torch.cat([x_node, static_feats], dim=1)  # [N,6]
        pred_delta = model(x_node, edge_index)
        du = pred_delta[:, 0]
        dv = pred_delta[:, 1]
        dp = pred_delta[:, 2]

        u_next = u_cur + du
        v_next = v_cur + dv
        p_next = p_cur + dp

        # Early stop on invalid values
        if (torch.isnan(u_next).any() or torch.isnan(v_next).any() or torch.isnan(p_next).any() or
            torch.isinf(u_next).any() or torch.isinf(v_next).any() or torch.isinf(p_next).any()):
            print(f"Stopping rollout early at step {i} due to non-finite values")
            break

        # Materialize arrays and append
        u_np = u_next.detach().cpu().numpy()
        v_np = v_next.detach().cpu().numpy()
        p_np = p_next.detach().cpu().numpy()
        step_out = base_step + i
        _append_rollout_step(out_csv_path, step_out, xs, ys, p_np, u_np, v_np)

        # Advance state
        u_cur = u_next
        v_cur = v_next
        p_cur = p_next

    print(f"Bootstrap rollout written to: {out_csv_path}")
    return str(out_csv_path)