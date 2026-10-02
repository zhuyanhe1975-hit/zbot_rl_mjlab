"""Interactive Zbot standing-pose and coordinate-frame visualizer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from .task_card import DEFAULT_JOINT_POS

ASSET_PATH = Path(__file__).resolve().parents[2] / "assets/zbot_6dof/robot_geometry_inertia.xml"


def standing_data(asset_path: Path = ASSET_PATH) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    model = mujoco.MjModel.from_xml_path(str(asset_path))
    data = mujoco.MjData(model)
    data.qpos[:3] = (0.0, -0.06, 0.0)
    data.qpos[7 : 7 + len(DEFAULT_JOINT_POS)] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    return model, data, base_id


def walking_frame(rotation: np.ndarray) -> dict[str, np.ndarray]:
    """Compute locomotion axes with world gravity as the vertical reference.

    The base body's local ``+Z`` is a lateral axis on this asset, so it must not
    be used as the up direction.  We use its world projection to define lateral
    direction, then derive forward from gravity and lateral.
    """
    down_w = np.array([0.0, 0.0, -1.0])
    up_w = -down_w
    lateral_w = rotation[:, 2].copy()
    lateral_w[2] = 0.0
    lateral_w /= np.linalg.norm(lateral_w)
    forward_w = np.cross(down_w, lateral_w)
    forward_w /= np.linalg.norm(forward_w)
    left_w = np.cross(up_w, forward_w)
    left_w /= np.linalg.norm(left_w)
    return {
        "forward": forward_w,
        "backward": -forward_w,
        "left": left_w,
        "right": -left_w,
        "up": up_w,
        "down": down_w,
    }


def print_pose(model: mujoco.MjModel, data: mujoco.MjData, base_id: int) -> None:
    rotation = data.xmat[base_id].reshape(3, 3)
    vectors = walking_frame(rotation)
    print(f"asset: {ASSET_PATH}")
    print(f"base position: {np.array2string(data.xpos[base_id], precision=5)}")
    print("base axes (world):")
    print(rotation)
    for name, vector in vectors.items():
        print(f"computed {name:7s}: {np.array2string(vector, precision=5)}")
    print("joint axes (world):")
    for joint_id in range(1, model.njnt):
        name = model.joint(joint_id).name
        print(f"  {name:8s} anchor={data.xanchor[joint_id]} axis={data.xaxis[joint_id]}")


def add_arrow(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray, color: tuple[float, ...], width: float = 0.008) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(color),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
    scene.ngeom += 1


def add_visuals(viewer, model: mujoco.MjModel, data: mujoco.MjData, base_id: int) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    rotation = data.xmat[base_id].reshape(3, 3)
    origin = data.xpos[base_id].copy()
    length = 0.30
    # Base frame: X red, Y green, Z blue.
    for axis, color in zip(rotation.T, ((1, 0, 0, 1), (0, 1, 0, 1), (0, 0.4, 1, 1))):
        add_arrow(scene, origin, origin + length * axis, color, width=0.004)
    # Calculated locomotion frame: forward/back cyan, left/right yellow,
    # gravity-up/down green/magenta.
    vectors = walking_frame(rotation)
    for name, color in (
        ("forward", (0, 1, 1, 1)),
        ("backward", (0, 0.5, 0.7, 1)),
        ("left", (1, 1, 0, 1)),
        ("right", (0.8, 0.5, 0, 1)),
        ("up", (0, 1, 0, 1)),
        ("down", (1, 0, 1, 1)),
    ):
        add_arrow(scene, origin, origin + length * vectors[name], color, width=0.005)
    # Joint axes: white, with a short arrow centered at each hinge anchor.
    for joint_id in range(1, model.njnt):
        anchor = data.xanchor[joint_id]
        axis = data.xaxis[joint_id]
        add_arrow(scene, anchor - 0.09 * axis, anchor + 0.09 * axis, (1, 1, 1, 1), width=0.003)
    # Inertia principal axes: thin pastel arrows at each body's COM.
    inertia_colors = ((1, 0.3, 0.3, 0.9), (0.3, 1, 0.3, 0.9), (0.3, 0.6, 1, 0.9))
    for body_id in range(1, model.nbody):
        local_rotation_flat = np.empty(9)
        mujoco.mju_quat2Mat(local_rotation_flat, model.body(body_id).iquat)
        local_rotation = local_rotation_flat.reshape(3, 3)
        inertia_rotation = data.xmat[body_id].reshape(3, 3) @ local_rotation
        com = data.xipos[body_id]
        for axis, color in zip(inertia_rotation.T, inertia_colors):
            add_arrow(scene, com, com + 0.12 * axis, color, width=0.002)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-only", action="store_true", help="Print frame diagnostics without opening a window")
    parser.add_argument("--inertia-source", choices=("usd", "geometry"), default="geometry", help="Select authored USD inertia or geometry-derived inertia")
    args = parser.parse_args()
    asset_path = ASSET_PATH if args.inertia_source == "usd" else ASSET_PATH.with_name("robot_geometry_inertia.xml")
    print(f"Selected inertia source: {args.inertia_source}; model: {asset_path}")
    model, data, base_id = standing_data(asset_path)
    print_pose(model, data, base_id)
    if args.print_only:
        return
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo viewer is unavailable in the selected MJLab environment") from exc
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = data.xpos[base_id]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        while viewer.is_running():
            add_visuals(viewer, model, data, base_id)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
