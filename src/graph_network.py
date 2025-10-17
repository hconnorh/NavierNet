import os
import csv
import glob
import json
import numpy as np
import pyvista as pv
import polars as pl

from typing import Dict, Tuple, List, Optional
from pathlib import Path

from extract_training_data import _as_point_data

ROOT = Path(__file__).resolve().parent.parent

COORD_DECIMALS_PRIMARY = 8
COORD_DECIMALS_FALLBACK_1 = 6
COORD_DECIMALS_FALLBACK_2 = 4

def _round_xy(points_xy: np.ndarray, decimals: int = COORD_DECIMALS_PRIMARY) -> np.ndarray:
    """
    Round xy coordinates to stabilize float-based lookups.
    """
    out = np.empty_like(points_xy, dtype=float)
    out[:, 0] = np.round(points_xy[:, 0].astype(float), decimals)
    out[:, 1] = np.round(points_xy[:, 1].astype(float), decimals)
    return out


def _pick_first_case(results_root: str) -> str:
    cases = [d for d in sorted(os.listdir(results_root)) if os.path.isdir(os.path.join(results_root, d))]
    if not cases:
        raise SystemExit(f"No cases found under {results_root}")
    return cases[0]


def _load_first_vtu(sim_name: str, case_name: str) -> pv.DataSet:
    vtu_dir = ROOT / "sims" / sim_name / "pyfr_results" / case_name
    vtus = sorted(glob.glob(os.path.join(vtu_dir, "*.vtu")))
    if not vtus:
        raise SystemExit(f"No .vtu files found in {vtu_dir}")
    mesh = pv.read(vtus[0])
    return _as_point_data(mesh)


def _build_boundary_flags(mesh: pv.DataSet, ref_map: Dict[Tuple[float, float], int]) -> np.ndarray:
    """
    Mark points that lie on the domain boundary as 1, else 0, aligned to the
    CSV-derived node ordering represented by `ref_map`.

    We must size the flags array to the number of nodes in `ref_map` (the
    canonical ordering used for node_x/node_y), not to the raw mesh point
    count; the CSV may have been deduplicated.
    """
    # Determine canonical node count from mapped indices (not number of keys)
    num_nodes = (max(ref_map.values()) + 1) if ref_map else 0
    try:
        boundary = mesh.extract_feature_edges(boundary_edges=True)
    except Exception:
        # Fallback: no boundary detection → zeros sized to canonical node count
        return np.zeros(num_nodes, dtype=np.int8)

    # Allocate flags to match CSV node ordering length
    flags = np.zeros(num_nodes, dtype=np.int8)
    if boundary is None or boundary.n_points == 0:
        return flags

    bxy = _round_xy(np.asarray(boundary.points)[:, :2])
    for x, y in bxy:
        key = (float(x), float(y))
        if key in ref_map:
            flags[ref_map[key]] = 1
    return flags


def _extract_edges(mesh: pv.DataSet, ref_map: Dict[Tuple[float, float], int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract undirected edges from the mesh using VTK's edge extractor and map
    them to node indices via coordinate rounding. Returns directed edge_index (2,E)
    and edge_attr (E,3) with columns [dx, dy, dist].
    """
    # Use extract_all_edges for UnstructuredGrid compatibility
    edges_poly = mesh.extract_all_edges()
    if edges_poly.n_cells == 0:
        raise SystemExit("No edges found from extract_all_edges(); mesh may be invalid")

    # Build coordinate arrays for node attributes
    ref_points = np.asarray(mesh.points)[:, :2].astype(float)

    # Parse VTK polyline connectivity: [n, id0, id1, n, id2, id3, ...]
    lines = edges_poly.lines
    pts = np.asarray(edges_poly.points)[:, :2].astype(float)

    # Map edge endpoints (in edges_poly local indexing) back to ref node indices via coordinates
    def map_local_id(local_id: int) -> int:
        x, y = pts[local_id]
        # Try progressively coarser rounding to mitigate tiny numeric mismatches
        for dec in (COORD_DECIMALS_PRIMARY, COORD_DECIMALS_FALLBACK_1, COORD_DECIMALS_FALLBACK_2):
            key = (float(round(x, dec)), float(round(y, dec)))
            if key in ref_map:
                return ref_map[key]
        raise KeyError(f"Edge point not found in reference map for coordinate: ({x}, {y})")

    undirected: set[Tuple[int, int]] = set()
    i = 0
    L = int(lines.size)
    while i < L:
        n = int(lines[i])
        ids = lines[i + 1 : i + 1 + n]
        # Expect n == 2 for lines; handle polylines by connecting consecutive ids
        if n >= 2:
            for a, b in zip(ids[:-1], ids[1:]):
                ia = map_local_id(int(a))
                ib = map_local_id(int(b))
                if ia == ib:
                    continue
                # store undirected unique
                if ia < ib:
                    undirected.add((ia, ib))
                else:
                    undirected.add((ib, ia))
        i += 1 + n

    if not undirected:
        raise SystemExit("Failed to build any mesh edges from VTK lines")

    # Build directed edge_index and edge_attr
    xs = ref_points[:, 0]
    ys = ref_points[:, 1]
    src_list: List[int] = []
    dst_list: List[int] = []
    attrs: List[Tuple[float, float, float]] = []

    for ia, ib in undirected:
        dx = float(xs[ib] - xs[ia])
        dy = float(ys[ib] - ys[ia])
        dist = float(np.hypot(dx, dy))

        # ia -> ib
        src_list.append(ia)
        dst_list.append(ib)
        attrs.append((dx, dy, dist))

        # ib -> ia
        src_list.append(ib)
        dst_list.append(ia)
        attrs.append((-dx, -dy, dist))

    edge_index = np.asarray([src_list, dst_list], dtype=np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32)
    return edge_index, edge_attr


def _get_first_step_value(results_csv: str) -> int:
    df = pl.read_csv(results_csv, columns=["step"], n_rows=1)
    if df.height == 0:
        raise SystemExit(f"CSV appears empty: {results_csv}")
    val = df.get_column("step")[0]
    return int(val)


def _load_csv_ref_points(results_csv: str, expected_count: Optional[int]) -> Tuple[np.ndarray, Dict[Tuple[float, float], int]]:
    """
    Load (n_x, n_y) for the first timestep only as the canonical node ordering.
    Returns (points_xy, ref_map) where ref_map uses multiple rounding levels to reduce
    numeric mismatch issues.
    """
    s0 = _get_first_step_value(results_csv)

    scan = (pl.scan_csv(results_csv)
              .with_columns(pl.col("step").cast(pl.Int64))
              .filter(pl.col("step") == pl.lit(s0))
              .select(["n_x", "n_y"]))

    if expected_count is not None:
        scan = scan.limit(expected_count)

    df = scan.collect(engine="auto")
    if df.height == 0:
        raise SystemExit(f"Failed to load step=={s0} rows from {os.path.basename(results_csv)}")

    xs = df.get_column("n_x").to_numpy()
    ys = df.get_column("n_y").to_numpy()
    pts = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])

    # Build multi-precision maps
    maps: List[Dict[Tuple[float, float], int]] = []
    for dec in (COORD_DECIMALS_PRIMARY, COORD_DECIMALS_FALLBACK_1, COORD_DECIMALS_FALLBACK_2):
        keys = [(float(round(x, dec)), float(round(y, dec))) for x, y in pts]
        maps.append({k: i for i, k in enumerate(keys)})

    # Merge into a single ref_map preferring higher precision entries
    ref_map: Dict[Tuple[float, float], int] = {}
    for m in reversed(maps):  # start from coarsest to ensure finer overwrites
        ref_map.update(m)
    return pts, ref_map


def _align_csv_timeseries(sim_name: str, case_name: str, ref_map: Dict[Tuple[float, float], int], num_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load `<case>-results.csv` and align per-timestep fields using row order.
    Any timestep with a row count != `num_nodes` is skipped to keep arrays consistent.

    Returns (steps, U, V, P) with shapes:
      - steps: (T_kept,)
      - U, V, P: (T_kept, N)
    """
    td_dir = ROOT / "sims" / sim_name / "training_data"
    results_csv = td_dir / f"{case_name}-results.csv"
    if not results_csv.exists():
        matches = sorted(td_dir.glob("*-results.csv"))
        if not matches:
            raise SystemExit(f"Could not find results CSV under {td_dir}")
        results_csv = matches[0]  # take first match

    steps: List[int] = []
    U_rows: List[np.ndarray] = []
    V_rows: List[np.ndarray] = []
    P_rows: List[np.ndarray] = []

    with open(results_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Expect: step,n_x,n_y,p,u,v,vn
        idx_step = header.index("step")
        idx_p = header.index("p")
        idx_u = header.index("u")
        idx_v = header.index("v")

        current_step: Optional[int] = None
        cur_u = None
        cur_v = None
        cur_p = None
        row_idx = 0

        def flush_step():
            nonlocal current_step, cur_u, cur_v, cur_p, row_idx
            if current_step is None:
                return
            if row_idx == num_nodes:
                steps.append(current_step)
                U_rows.append(cur_u)
                V_rows.append(cur_v)
                P_rows.append(cur_p)
            # else: skip incomplete step silently
            current_step = None
            cur_u = cur_v = cur_p = None
            row_idx = 0

        for row in reader:
            s = int(float(row[idx_step]))
            if current_step is None:
                current_step = s
                cur_u = np.empty((num_nodes,), dtype=np.float32)
                cur_v = np.empty((num_nodes,), dtype=np.float32)
                cur_p = np.empty((num_nodes,), dtype=np.float32)
                row_idx = 0
            elif s != current_step:
                flush_step()
                current_step = s
                cur_u = np.empty((num_nodes,), dtype=np.float32)
                cur_v = np.empty((num_nodes,), dtype=np.float32)
                cur_p = np.empty((num_nodes,), dtype=np.float32)
                row_idx = 0

            # Fill in by row order
            cur_p[row_idx] = float(row[idx_p])
            cur_u[row_idx] = float(row[idx_u])
            cur_v[row_idx] = float(row[idx_v])
            row_idx += 1

        # Flush last step
        if current_step is not None:
            flush_step()

    steps_arr = np.asarray(steps, dtype=int)
    U = np.stack(U_rows, axis=0)
    V = np.stack(V_rows, axis=0)
    P = np.stack(P_rows, axis=0)
    return steps_arr, U, V, P


def build_graph(sim_name: str, case_name: Optional[str] = None) -> str:
    """
    Build a static graph from the first .vtu of a case and align CSV fields
    across timesteps to the node ordering.

    Saves artifacts under `sims/<sim_name>/training_data/graph/<case_name>/`:
      - graph.npz: node_x, node_y, node_flags, edge_index, edge_attr
      - timeseries.npz: steps, U, V, P, dU, dV, dP
      - meta.json: basic metadata

    Returns the output directory path.
    """
    sim_dir = ROOT / "sims" / sim_name
    results_root = sim_dir / "pyfr_results"
    if not os.path.isdir(results_root):
        raise SystemExit(f"Results root not found: {results_root}")

    if case_name is None:
        case_name = _pick_first_case(results_root)

    print(f"Loading first .vtu for {case_name}")
    mesh = _load_first_vtu(sim_name, case_name)
    mesh_points = np.asarray(mesh.points)[:, :2].astype(float)
    N_mesh = mesh_points.shape[0]

    print(f"Building node ordering from CSV for {case_name}")
    # Build node ordering from CSV (canonical for features/targets), then map mesh entities to it
    td_dir = os.path.join(sim_dir, "training_data")
    results_csv = os.path.join(td_dir, f"{case_name}-results.csv")
    if not os.path.isfile(results_csv):
        matches = sorted(glob.glob(os.path.join(td_dir, "*-results.csv")))
        if not matches:
            raise SystemExit(f"Could not find results CSV under {td_dir}")
        results_csv = matches[0]

    ref_points, ref_map = _load_csv_ref_points(results_csv, expected_count=N_mesh)
    N = ref_points.shape[0]

    print(f"Building boundary flags for {case_name}")
    # Flags from mesh boundaries mapped into CSV node ordering
    node_flags = _build_boundary_flags(mesh, ref_map)

    print(f"Building edges and attributes for {case_name}")
    # Edges and attributes
    edge_index, edge_attr = _extract_edges(mesh, ref_map)

    print(f"Aligning timeseries for {case_name}")
    # Timeseries alignment
    steps, U, V, P = _align_csv_timeseries(sim_name, case_name, ref_map, N)
    dU = (U[1:] - U[:-1]).astype(np.float32)
    dV = (V[1:] - V[:-1]).astype(np.float32)
    dP = (P[1:] - P[:-1]).astype(np.float32)

    print(f"Saving artifacts for {case_name}")
    # Persist
    out_dir = os.path.join(sim_dir, "training_data", "graph", case_name)
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, "graph.npz"),
        node_x=ref_points[:, 0].astype(np.float32),
        node_y=ref_points[:, 1].astype(np.float32),
        node_flags=node_flags.astype(np.int8),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    np.savez_compressed(
        os.path.join(out_dir, "timeseries.npz"),
        steps=steps.astype(np.int32),
        U=U,
        V=V,
        P=P,
        dU=dU,
        dV=dV,
        dP=dP,
    )

    meta = {
        "sim_name": sim_name,
        "case_name": case_name,
        "num_nodes": int(N),
        "num_edges_directed": int(edge_index.shape[1]),
        "num_steps": int(steps.size),
        "features": ["x", "y", "flag", "u", "v", "p"],
        "edge_attr": ["dx", "dy", "dist"],
        "targets": ["dU", "dV", "dP"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


if __name__ == "__main__":
    # Parameters
    SIM_NAME = "example-model"
    CASE_NAME = 'case0'

    # Build graph architecture
    out = build_graph(SIM_NAME, CASE_NAME)
    print(f"Graph dataset written to: {out}")


