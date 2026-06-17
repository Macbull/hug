"""Convert HUG grasp predictions into robot-facing targets.

This keeps HUG as a grasp-perception front end: it reads predicted MANO grasps
saved under ``grasp_pred/``, applies a camera->robot calibration transform,
builds a simple pre-grasp target, runs basic safety filters, and exports a
robot-ready JSON bundle for downstream arm/hand controllers.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from rich.console import Console

console = Console()

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_transform(path: Path) -> np.ndarray:
    """Load a 4x4 homogeneous transform from txt/npy/json."""
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            for key in ("T_base_camera", "transform", "matrix"):
                if key in raw:
                    raw = raw[key]
                    break
        data = np.asarray(raw, dtype=np.float64)
    else:
        data = np.loadtxt(path)
    data = np.asarray(data, dtype=np.float64)
    if data.shape == (4, 4):
        return data
    if data.shape == (3, 4):
        return np.vstack([data, np.array([0.0, 0.0, 0.0, 1.0])])
    if data.size == 16:
        return data.reshape(4, 4)
    raise ValueError(f"{path}: expected a 4x4 or 3x4 transform, got {data.shape}")


def _resolve_predictions(
    dataset_path: Path, prediction_path: Optional[Path]
) -> list[Path]:
    """Resolve one or more prediction pkls."""
    if prediction_path is None:
        prediction_path = dataset_path / "grasp_pred"
    if prediction_path.is_file():
        return [prediction_path]
    if prediction_path.is_dir():
        return sorted(prediction_path.rglob("*.pkl"))
    raise FileNotFoundError(prediction_path)


def _camera_dict(data: dict) -> dict:
    camera = data["camera"]
    return camera if isinstance(camera, dict) else camera.__dict__


def _decode_depth_m(depth_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(depth_bytes, np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Failed to decode depth image")
    depth = depth.astype(np.float32)
    depth[depth >= 65535] = 0
    return depth / 1000.0


def _decode_condition_point_uv(
    data: dict, width: int, height: int
) -> Optional[np.ndarray]:
    stored = data.get("condition_point")
    if stored is not None:
        uv = np.asarray(stored, dtype=np.float32).reshape(2)
        return uv
    point_bytes = data.get("object_mask", b"") or b""
    if len(point_bytes) == 8:
        uv_norm = np.frombuffer(point_bytes, dtype=np.float32).copy()
        return np.array(
            [uv_norm[0] * width, uv_norm[1] * height], dtype=np.float32
        )
    return None


def _pixel_to_xyz(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array(
        [(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=np.float32
    )


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homo = np.concatenate(
        [pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (homo @ T.T)[:, :3].astype(np.float32)


def _offset_pose(T: np.ndarray, offset_local: np.ndarray) -> np.ndarray:
    T_out = np.array(T, dtype=np.float64, copy=True)
    T_out[:3, 3] = T[:3, :3] @ offset_local + T[:3, 3]
    return T_out


def _workspace_mask(
    points: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return np.all((pts >= workspace_min) & (pts <= workspace_max), axis=1)


def _ranking_scores(
    wrist_base: np.ndarray,
    pregrasp_base: np.ndarray,
    landmarks_base: np.ndarray,
    fingertips_base: np.ndarray,
    object_point_base: Optional[np.ndarray],
    safety: dict,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
) -> dict:
    workspace_center = 0.5 * (workspace_min + workspace_max)
    workspace_radius = max(
        np.linalg.norm(workspace_max - workspace_min) * 0.5, 1e-6
    )
    reach_dist = 0.5 * (
        np.linalg.norm(wrist_base - workspace_center)
        + np.linalg.norm(pregrasp_base - workspace_center)
    )
    reachability = float(np.clip(1.0 - reach_dist / workspace_radius, 0.0, 1.0))

    if object_point_base is not None:
        tip_centroid = fingertips_base.mean(axis=0)
        palm = landmarks_base[0]
        tip_dist = np.linalg.norm(tip_centroid - object_point_base)
        palm_dist = np.linalg.norm(palm - object_point_base)
        contact_quality = float(
            np.clip(1.0 - (0.7 * tip_dist + 0.3 * palm_dist) / 0.15, 0.0, 1.0)
        )
    else:
        contact_quality = 0.0

    safety_score = float(np.mean(list(safety.values())))
    total = 0.5 * reachability + 0.3 * contact_quality + 0.2 * safety_score
    if not safety["wrist_in_workspace"]:
        total *= 0.25
    if not safety["pregrasp_in_workspace"]:
        total *= 0.5
    if not safety["camera_frame_valid"]:
        total = 0.0
    return {
        "reachability": reachability,
        "contact_quality": contact_quality,
        "safety": safety_score,
        "total": float(total),
    }


def _build_robot_target(
    pred_path: Path,
    T_base_camera: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    table_height: float,
    pregrasp_offset: np.ndarray,
    include_mesh: bool,
) -> dict:
    data = _load_pickle(pred_path)
    grasp = data.get("grasp")
    if grasp is None:
        raise ValueError(f"{pred_path} does not contain a predicted grasp")

    camera = _camera_dict(data)
    K = np.asarray(camera["K"], dtype=np.float32)
    width = int(camera["width"])
    height = int(camera["height"])

    T_camera_wrist = np.asarray(grasp["T_camera_wrist"], dtype=np.float64)
    T_base_wrist = T_base_camera @ T_camera_wrist
    T_base_pregrasp = _offset_pose(T_base_wrist, pregrasp_offset)

    landmarks_camera = np.asarray(grasp["landmarks_3d"], dtype=np.float32)
    landmarks_base = _transform_points(T_base_camera, landmarks_camera)
    fingertips_base = landmarks_base[FINGERTIP_INDICES]
    mesh_vertices_camera = np.asarray(grasp["mesh_vertices"], dtype=np.float32)
    mesh_vertices_base = _transform_points(T_base_camera, mesh_vertices_camera)

    condition_uv = _decode_condition_point_uv(data, width, height)
    object_point_camera = None
    object_point_base = None
    if condition_uv is not None and data.get("depth"):
        depth_m = _decode_depth_m(data["depth"])
        u = float(np.clip(condition_uv[0], 0, width - 1))
        v = float(np.clip(condition_uv[1], 0, height - 1))
        ui = int(np.clip(round(u), 0, width - 1))
        vi = int(np.clip(round(v), 0, height - 1))
        d = float(depth_m[vi, ui])
        if d > 0:
            object_point_camera = _pixel_to_xyz(u, v, d, K)
            object_point_base = _transform_points(T_base_camera, object_point_camera)[0]

    wrist_base = T_base_wrist[:3, 3].astype(np.float32)
    pregrasp_base = T_base_pregrasp[:3, 3].astype(np.float32)
    safety = {
        "camera_frame_valid": bool(T_camera_wrist[2, 3] > 0.0),
        "wrist_in_workspace": bool(
            _workspace_mask(wrist_base, workspace_min, workspace_max)[0]
        ),
        "pregrasp_in_workspace": bool(
            _workspace_mask(pregrasp_base, workspace_min, workspace_max)[0]
        ),
        "fingertips_in_workspace": bool(
            _workspace_mask(fingertips_base, workspace_min, workspace_max).all()
        ),
        "wrist_above_table": bool(wrist_base[2] >= table_height),
        "fingertips_above_table": bool(np.all(fingertips_base[:, 2] >= table_height)),
    }
    if object_point_base is not None:
        safety["object_point_in_workspace"] = bool(
            _workspace_mask(object_point_base, workspace_min, workspace_max)[0]
        )
        safety["object_point_above_table"] = bool(object_point_base[2] >= table_height)

    ranking = _ranking_scores(
        wrist_base,
        pregrasp_base,
        landmarks_base,
        fingertips_base,
        object_point_base,
        safety,
        workspace_min,
        workspace_max,
    )

    result = {
        "prediction_path": str(pred_path),
        "condition_point_uv": (
            condition_uv.tolist() if condition_uv is not None else None
        ),
        "object_point_camera_m": (
            object_point_camera.astype(np.float32).tolist()
            if object_point_camera is not None
            else None
        ),
        "object_point_base_m": (
            object_point_base.astype(np.float32).tolist()
            if object_point_base is not None
            else None
        ),
        "T_camera_wrist": T_camera_wrist.astype(np.float32).tolist(),
        "T_base_wrist": T_base_wrist.astype(np.float32).tolist(),
        "T_base_pregrasp": T_base_pregrasp.astype(np.float32).tolist(),
        "landmarks_camera_m": landmarks_camera.tolist(),
        "landmarks_base_m": landmarks_base.tolist(),
        "fingertips_base_m": fingertips_base.tolist(),
        "safety": safety,
        "ranking": ranking,
    }
    if include_mesh:
        result["mesh_vertices_camera_m"] = mesh_vertices_camera.tolist()
        result["mesh_vertices_base_m"] = mesh_vertices_base.tolist()
    return result


def main(
    dataset_path: Path,
    calibration_path: Path,
    prediction_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    workspace_min: tuple[float, float, float] = (-0.8, -0.6, 0.0),
    workspace_max: tuple[float, float, float] = (0.8, 0.6, 1.2),
    table_height: float = 0.0,
    pregrasp_offset: tuple[float, float, float] = (0.05, 0.05, 0.0),
    include_mesh: bool = True,
    top_k: Optional[int] = None,
) -> None:
    """Export calibrated robot targets from saved HUG predictions.

    Args:
        dataset_path: Prepared dataset root. Defaults prediction discovery to
            ``dataset_path/grasp_pred``.
        calibration_path: 4x4 ``T_base_camera`` transform in txt/npy/json form.
        prediction_path: Prediction pkl or directory. If omitted, uses
            ``dataset_path/grasp_pred``.
        output_path: Output JSON file. Defaults to ``robot_targets.json`` at the
            dataset root, or ``<prediction>.robot.json`` for a single input file.
        workspace_min: Inclusive XYZ lower bound in robot-base meters.
        workspace_max: Inclusive XYZ upper bound in robot-base meters.
        table_height: Minimum allowed Z in robot-base meters.
        pregrasp_offset: Wrist-local XYZ offset in meters for the pre-grasp pose.
        include_mesh: Include full MANO mesh vertices in the exported JSON.
        top_k: Keep only the top-k ranked targets after filtering.
    """
    dataset_path = Path(dataset_path)
    predictions = _resolve_predictions(dataset_path, prediction_path)
    T_base_camera = _load_transform(calibration_path)
    workspace_min_np = np.asarray(workspace_min, dtype=np.float32)
    workspace_max_np = np.asarray(workspace_max, dtype=np.float32)
    pregrasp_offset_np = np.asarray(pregrasp_offset, dtype=np.float32)

    targets = [
        _build_robot_target(
            pred_path,
            T_base_camera,
            workspace_min_np,
            workspace_max_np,
            table_height,
            pregrasp_offset_np,
            include_mesh,
        )
        for pred_path in predictions
    ]
    targets.sort(key=lambda item: item["ranking"]["total"], reverse=True)
    if top_k is not None:
        targets = targets[: max(top_k, 0)]

    if output_path is None:
        if len(predictions) == 1 and predictions[0].is_file():
            output_path = predictions[0].with_suffix(".robot.json")
        else:
            output_path = dataset_path / "robot_targets.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "T_base_camera": T_base_camera.astype(np.float32).tolist(),
        "workspace_min": workspace_min_np.tolist(),
        "workspace_max": workspace_max_np.tolist(),
        "table_height": table_height,
        "pregrasp_offset": pregrasp_offset_np.tolist(),
        "targets": targets,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]wrote {output_path}[/green]")
    console.print(f"[cyan]exported {len(targets)} target(s)[/cyan]")


if __name__ == "__main__":
    tyro.cli(main)
