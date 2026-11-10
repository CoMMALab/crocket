"""cuRobo's kinematics on the benchmark URDF with the URDF's own sphere set (not cuRobo's yml
spheres), so it places exactly the spheres the other methods place."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def root_link(urdf) -> str:
    tree = ET.parse(urdf).getroot()
    children = {c.get("link") for c in tree.iter("child")}
    return next(l.get("name") for l in tree.iter("link") if l.get("name") not in children)


def kinematics_dict(urdf, tool_frames=None) -> dict:
    """cuRobo kinematics config for ``urdf`` with the URDF's own spheres."""
    links, centres, radii = common.spheres(urdf)
    spheres: dict[str, list] = {}
    for l, c, r in zip(links, centres, radii):
        spheres.setdefault(l, []).append({"center": c.tolist(), "radius": float(r)})
    return {
        "base_link": root_link(urdf), "urdf_path": str(urdf),
        "collision_link_names": list(spheres), "collision_spheres": spheres,
        "tool_frames": tool_frames or [links[-1]],
        "self_collision_buffer": {}, "self_collision_ignore": {},  # unused here, required
    }


def curobo_model(robot: str, names: list[str], urdf=None, tool_frames=None,
                 jacobian: bool = False):
    """(Kinematics, perm): ``q[:, perm]`` is cuRobo's joint order. cuRobo keeps only joints
    that move a sphere or tool frame, which leaves the spheres unchanged."""
    from curobo.kinematics import Kinematics, KinematicsCfg
    cfg = KinematicsCfg.from_data_dict(kinematics_dict(urdf or common.FK_URDF[robot], tool_frames))
    model = Kinematics(cfg, compute_jacobian=jacobian, compute_spheres=True)
    perm = [names.index(n) for n in model.joint_names]
    return model, perm
