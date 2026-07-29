"""Scripted geometric push expert for PushT.

The controller exploits two facts about this env:
  - the pusher is a KINEMATIC circle (its motion is exactly what we command), and
  - space damping is 0, so the block is fully quasi-static: it moves only while it
    is actively being pushed and stops the instant contact ends.

So a push is just: pick where to push, steer the pusher to a stand-off point just
outside that face (routing around a safe ring whenever the straight path would clip
the T), then press in. Because the pusher never crosses the T footprint, it avoids
the "blue circle ends up on top of the T" artifact of the random collector, and the
straight, velocity-limited moves give the smooth trajectories the random walk lacked.

Position is corrected by pushing the trailing face through the CoM (near-zero torque
translation); orientation by pushing a stem tip sideways -- a long lever that spins
the block nearly in place. The two are interleaved with hysteresis so they don't
fight.
"""

from __future__ import annotations

import numpy as np
from stable_worldmodel.envs.pusht.env import PushT


# T block outline in the block's local frame (see env.add_tee): a bar across the
# top (y in [0,30], x in [-60,60]) and a stem hanging below (y in [30,120],
# x in [-15,15]). Consecutive vertices form the faces the expert pushes on.
TEE_OUTLINE_LOCAL = np.array(
    [(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)],
    dtype=np.float64,
)
# Contact points for rotating the T: pushing a stem-tip side face sideways gives a
# long lever about the CoM, so the block spins nearly in place (verified: ~0.8 rad
# with <1px drift) instead of being dragged across the room like a bar-tip push.
# Each entry is (contact_local, outward_normal_local).
ROT_CONTACTS_LOCAL = [
    (np.array([15.0, 108.0]), np.array([1.0, 0.0])),   # right side of stem, near tip
    (np.array([-15.0, 108.0]), np.array([-1.0, 0.0])),  # left side of stem, near tip
]


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


def signed_angle_diff(target: float, cur: float) -> float:
    return (target - cur + np.pi) % (2 * np.pi) - np.pi


class ScriptedPushExpert:
    """Geometric quasi-static push controller for a single episode/env."""

    def __init__(self, env: PushT, max_action: float = 0.32, ring_margin: float = 14.0,
                 ang_tol: float = np.pi / 36):
        self.env = env
        self.max_action = max_action
        # Pymunk keeps agent shapes in a set; grab the circle radius.
        self.r = next(iter(env.agent.shapes)).radius
        max_reach = float(np.linalg.norm(TEE_OUTLINE_LOCAL - env.block.center_of_gravity, axis=1).max())
        self.r_ring = max_reach + self.r + ring_margin
        # Mode commitment. Thresholds are deliberately decoupled from the (tight)
        # success tolerance and generous, so a demo is essentially "rotate once, then
        # translate once" like the human PushT-M demos -- not a rotate/translate
        # ping-pong, where every switch costs a full trip around the T. Re-rotation
        # only happens if the angle drifts a lot, or as a single final cleanup once
        # the position is already on target (see _choose_contact).
        self.rot_trigger = max(0.18, 2.0 * ang_tol)   # (re)rotate only if clearly off
        self.rot_release = min(0.03, 0.5 * ang_tol)   # rotate tight so the one pass suffices
        self._mode = "translate"
        self._trans_edge = None  # local edge index we're committed to pushing
        self._rot_contact = None  # index into ROT_CONTACTS_LOCAL we're committed to

    def _world(self, local_pt: np.ndarray) -> np.ndarray:
        return np.array(self.env.block.local_to_world(tuple(local_pt)))

    def _inside_tee(self, world_pt: np.ndarray, pad: float) -> bool:
        """Is a world point within the T footprint (bar or stem) inflated by pad?"""
        x, y = self.env.block.world_to_local(tuple(world_pt))
        in_bar = -60 - pad <= x <= 60 + pad and 0 - pad <= y <= 30 + pad
        in_stem = -15 - pad <= x <= 15 + pad and 30 - pad <= y <= 120 + pad
        return in_bar or in_stem

    def _blocked(self, p: np.ndarray, q: np.ndarray) -> bool:
        """Would the straight pusher move p->q clip the T? (pad by the pusher radius)."""
        pad = self.r * 0.8
        for f in np.linspace(0.0, 1.0, 12):
            if self._inside_tee(p + f * (q - p), pad):
                return True
        return False

    def _edge_contact(self, com: np.ndarray, i: int):
        """Perpendicular-through-CoM contact (foot, outward_normal) for local edge i."""
        n_edges = len(TEE_OUTLINE_LOCAL)
        p1 = self._world(TEE_OUTLINE_LOCAL[i])
        p2 = self._world(TEE_OUTLINE_LOCAL[(i + 1) % n_edges])
        edge = p2 - p1
        length = np.linalg.norm(edge)
        n = unit(np.array([edge[1], -edge[0]]))
        if np.dot(n, p1 + 0.5 * edge - com) < 0:  # point outward
            n = -n
        s = np.clip(np.dot(com - p1, edge) / length**2, 0.0, 1.0)
        return p1 + s * edge, n

    def _translate_contact(self, com: np.ndarray, u: np.ndarray):
        """Face to push (perpendicular, through the CoM) to translate the block along u.

        Picking the back face and contacting at the foot of the perpendicular from
        the CoM keeps the push force line through the CoM, so the block slides
        along u with near-zero torque instead of spinning. The committed face is
        kept while it still faces away from the motion, so the pusher pushes
        instead of endlessly re-orbiting between near-equal faces.
        """
        def score(i):
            foot, n = self._edge_contact(com, i)
            offset = np.linalg.norm((foot - com) - np.dot(foot - com, n) * n)  # lever arm
            return np.dot(n, -u) - 0.01 * offset, foot, n

        if self._trans_edge is not None:
            back, foot, n = score(self._trans_edge)
            if back > 0.2:  # still a valid back face -> keep pushing it
                return foot, n
        self._trans_edge = max(range(len(TEE_OUTLINE_LOCAL)), key=lambda i: score(i)[0])
        return self._edge_contact(com, self._trans_edge)

    def _choose_contact(self, com: np.ndarray, dp: np.ndarray, dth: float):
        """Return (contact_world, outward_world, push_dir_world) for this step's push."""
        # Rotate first (if the angle is off), then translate -- one trip each, like
        # the human demos. No end-game rotate/translate cleanup: once translating we
        # only re-rotate for a large drift, so the pusher doesn't dance around the T.
        if self._mode == "translate" and abs(dth) > self.rot_trigger:
            self._mode = "rotate"
            self._rot_contact = None
        elif self._mode == "rotate" and abs(dth) < self.rot_release:
            self._mode = "translate"
            self._trans_edge = None  # reselect the back face after rotating

        if self._mode == "rotate":
            # Rotate: commit to one bar-tip contact whose inward push torques the
            # block the right way, and keep it until the angle error flips sign,
            # so the pusher rotates the block instead of thrashing between tips.
            want = np.sign(dth)

            def rot_contact(idx):
                c_local, n_local = ROT_CONTACTS_LOCAL[idx]
                c = self._world(c_local)
                n = unit(self._world(c_local + n_local) - c)
                rvec = c - com
                torque = rvec[0] * (-n[1]) - rvec[1] * (-n[0])  # cross(r, push_dir=-n)
                return c, n, torque

            # Reuse the committed contact while it still torques the right way; else
            # pick the aligned contact that needs the least orbiting to reach.
            if self._rot_contact is not None and np.sign(rot_contact(self._rot_contact)[2]) == want:
                c, n, _ = rot_contact(self._rot_contact)
                return c, n, -n
            agent_bearing = unit(np.array(self.env.agent.position) - com)
            aligned = [i for i in range(len(ROT_CONTACTS_LOCAL)) if np.sign(rot_contact(i)[2]) == want]
            if aligned:
                self._rot_contact = max(aligned, key=lambda i: np.dot(unit(rot_contact(i)[0] - com), agent_bearing))
                c, n, _ = rot_contact(self._rot_contact)
                return c, n, -n
        # Translate: push perpendicular into the back face, through the CoM.
        u = unit(dp)
        foot, n = self._translate_contact(com, u)
        return foot, n, -n

    def target(self, agent: np.ndarray) -> np.ndarray:
        """World point to steer the pusher toward this step."""
        block = self.env.block
        com = np.array(block.local_to_world(block.center_of_gravity))
        bp = np.array(block.position)
        goal = self.env.goal_pose
        dp = goal[:2] - bp
        dth = signed_angle_diff(goal[2], block.angle)

        contact, outward, _push_dir = self._choose_contact(com, dp, dth)
        cur_bearing = unit(agent - com)
        self._speed_cap = self.max_action

        gap = 6.0
        stand_off = contact + outward * (self.r + gap)   # sit here before pressing
        press_pt = contact - outward * (self.r * 0.5)     # drive to here to push

        # In the pressing zone (in front of the face, outside it): push. This has to
        # be checked directly against the face -- the stand-off point is not radial
        # from the CoM for a stem-tip rotation push, so a CoM-bearing test misfires.
        to_face = float(np.dot(agent - contact, outward))
        lateral = np.linalg.norm((agent - contact) - to_face * outward)
        if -2.0 < to_face < self.r + gap + 6.0 and lateral < self.r:
            if self._mode == "translate":
                self._speed_cap = float(np.clip(0.012 * np.linalg.norm(dp), 0.06, self.max_action))
            else:  # rotate: slow down as the angle error closes
                self._speed_cap = float(np.clip(0.6 * abs(dth), 0.05, self.max_action))
            return press_pt

        # Otherwise steer to the stand-off point. Go straight when that path clears
        # the T; else route around the safe ring (never cut across the block).
        if not self._blocked(agent, stand_off):
            return stand_off
        if np.linalg.norm(agent - com) < self.r_ring - 6.0:
            return com + cur_bearing * self.r_ring  # radial move out is clear of the T
        target_bearing = unit(stand_off - com)
        step = self.max_action * self.env.action_scale / self.r_ring  # ~arc length cap
        cross = cur_bearing[0] * target_bearing[1] - cur_bearing[1] * target_bearing[0]
        cos = float(np.dot(cur_bearing, target_bearing))
        ang = np.clip(np.arctan2(cross, cos), -step, step)
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([c * cur_bearing[0] - s * cur_bearing[1],
                        s * cur_bearing[0] + c * cur_bearing[1]])
        return com + rot * self.r_ring

    def act(self, agent: np.ndarray) -> np.ndarray:
        delta = self.target(agent) - agent
        action = delta / self.env.action_scale
        norm = np.linalg.norm(action)
        if norm > self._speed_cap:
            action = action * (self._speed_cap / norm)
        return action.astype(np.float32)
