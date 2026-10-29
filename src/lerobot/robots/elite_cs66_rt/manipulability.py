# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Manipulability / singularity kinematics for the Elite CS66 drivers.

Pure-numpy forward kinematics, geometric Jacobian and a manipulability measure used to
*detect* proximity to any kinematic singularity (wrist ``q5≈0``, shoulder, elbow/boundary)
and smoothly damp the Cartesian servo command before the controller's internal IK spikes
joint velocity past its ``JOINT_IGNORE_SPEED`` bound and drops external control.

DH convention
-------------
The Elite controller uses **Modified (Craig) DH**. This must match
``third_party/libpyelite/vendor/elite_cs_sdk/plugin/kinematics/.../KdlKinematicsPlugin.cpp::setMDH``
exactly: each link is ``Rx(alpha_i) · Tx(a_i) · Tz(d_i) · Rz(theta_i)`` with the per-joint
parameters indexed as ``dh_alpha_[i], dh_a_[i], dh_d_[i]`` from ``KinematicsInfo``. A
standard-DH or mis-indexed builder silently yields ``w≡0`` at *every* configuration (not a
crash), which would freeze teleop — hence the offline UR5 regression test and the live
self-check.

Tool / frame invariance
-----------------------
The manipulability ``w = |det(J)|`` of the 6×6 geometric Jacobian is invariant to:
  * a rigid flange→TCP tool offset (``J_tcp = Shear · J_flange``, ``det(Shear)=1``), and
  * the base-mounting rotation (``blkdiag(R, R)``, ``det = det(R)^2 = 1``).
So ``w`` is computed from the **flange MDH chain and the current joint angles only** — the
unknown active TCP offset and the world↔base transform are irrelevant to the metric, and
``w`` vanishes at exactly the same joint configurations as the true TCP manipulability.
"""

from __future__ import annotations

import numpy as np

from lerobot.utils.rotation import Rotation

__all__ = [
    "mdh_link_transform",
    "fk_flange_frames",
    "fk_flange_pose",
    "flange_se3",
    "pose6_to_se3",
    "pose_delta",
    "tool_consistency",
    "geometric_jacobian",
    "manipulability",
    "damping_scale",
    "directional_scale",
    "joint_velocity_scale",
    "SELFCHECK_POS_TOL_M",
    "SELFCHECK_ROT_WARN_DEG",
]


def _rot_x(alpha: float) -> np.ndarray:
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _trans_x(a: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[0, 3] = a
    return t


def _trans_z(d: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[2, 3] = d
    return t


def mdh_link_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Modified-DH (Craig) link transform ``Rx(alpha)·Tx(a)·Tz(d)·Rz(theta)`` (4x4).

    Matches ``KdlKinematicsPlugin::setMDH``. ``theta`` is the joint angle (no offset —
    ``KinematicsInfo`` provides no theta offset; the live self-check validates this).
    """
    return _rot_x(alpha) @ _trans_x(a) @ _trans_z(d) @ _rot_z(theta)


def fk_flange_frames(dh_alpha, dh_a, dh_d, q) -> list[np.ndarray]:
    """Cumulative base→frame transforms ``[T0=I, T1, ..., Tn=flange]`` (each 4x4)."""
    frames = [np.eye(4, dtype=np.float64)]
    t = np.eye(4, dtype=np.float64)
    for i in range(len(q)):
        t = t @ mdh_link_transform(dh_alpha[i], dh_a[i], dh_d[i], q[i])
        frames.append(t.copy())
    return frames


def fk_flange_pose(dh_alpha, dh_a, dh_d, q) -> np.ndarray:
    """Flange pose ``[x, y, z, rx, ry, rz]`` (rotvec) in the arm base frame."""
    t = fk_flange_frames(dh_alpha, dh_a, dh_d, q)[-1]
    rotvec = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    return np.concatenate([t[:3, 3], rotvec])


def flange_se3(dh_alpha, dh_a, dh_d, q) -> np.ndarray:
    """Flange pose as a 4x4 SE(3) transform in the arm base frame."""
    return fk_flange_frames(dh_alpha, dh_a, dh_d, q)[-1]


def pose6_to_se3(pose6: np.ndarray) -> np.ndarray:
    """Convert a ``[x, y, z, rx, ry, rz]`` (rotvec) pose to a 4x4 SE(3) transform."""
    pose6 = np.asarray(pose6, dtype=np.float64)
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = pose6[:3]
    t[:3, :3] = Rotation.from_rotvec(pose6[3:6]).as_matrix()
    return t


def pose_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """6-vec Cartesian step ``a -> b``: ``[Δpos, rotvec of the relative rotation]``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rel = (Rotation.from_rotvec(b[3:6]) * Rotation.from_rotvec(a[3:6]).inv()).as_rotvec()
    return np.concatenate([b[:3] - a[:3], rel])


# FK self-check gate thresholds (shared by both drivers). The POSITION tolerance is the real gate:
# it validates the DH for the geometric Jacobian, which is all the guard's math depends on. The
# ROTATION value is only a logging threshold — a large tool-orientation drift is a benign
# about-flange-Z TCP-convention artifact that does NOT affect det(J) (see tool_consistency).
SELFCHECK_POS_TOL_M = 0.002  # max inferred-tool position drift across configs to trust the DH (m)
SELFCHECK_ROT_WARN_DEG = 2.0  # above this, log the (benign) tool-orientation convention drift


def tool_consistency(dh, q0, t0, q1, t1) -> tuple[float, float]:
    """Return ``(position_drift_m, rotation_drift_deg)`` of the inferred flange→TCP tool transform
    between two well-separated configs. ``dh`` is ``(alpha, a, d)``.

    The tool is one rigid body, so ``inv(flange_FK(q)) @ measured_TCP(q)`` must be the SAME
    transform at both configs. The two drift components validate DIFFERENT things, and the caller
    must treat them differently:

    * **Position drift** validates the DH itself — a wrong link length / ``alpha`` / joint sign makes
      the inferred tool ORIGIN inconsistent across configs. This is the part the singularity guard
      depends on: the geometric Jacobian (hence ``w = |det(J)|`` and the joint-velocity prediction)
      is a pure function of the DH's link positions and joint axes, so a position-consistent DH is a
      valid DH for the guard's math.
    * **Rotation drift** additionally folds in the TCP-orientation *convention*. On the live CS66 the
      controller's ``actual_TCP_pose`` carries a config-dependent twist about the flange Z axis
      (tool-mount / rotvec convention) that leaves the tool ORIGIN exactly consistent (a Z-aligned
      tool is invariant to Z rotation) yet shows up as a large rotation drift (18–42° observed
      2026-07-09). A rotation about the flange Z axis changes neither any joint axis ``z_i`` nor the
      end origin, so it does NOT enter ``det(J)``.

    Therefore the caller gates the guard on ``position_drift_m`` (``SELFCHECK_POS_TOL_M``) and treats
    ``rotation_drift_deg`` as a logged diagnostic, NOT a disable. See
    ``EliteCS66RT._setup_singularity_damping``.
    """
    tool0 = np.linalg.inv(flange_se3(*dh, q0)) @ pose6_to_se3(t0)
    tool1 = np.linalg.inv(flange_se3(*dh, q1)) @ pose6_to_se3(t1)
    pos_drift_m = float(np.linalg.norm(tool0[:3, 3] - tool1[:3, 3]))
    d_rot = tool0[:3, :3].T @ tool1[:3, :3]
    rot_drift_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(d_rot).as_rotvec())))
    return pos_drift_m, rot_drift_deg


def geometric_jacobian(dh_alpha, dh_a, dh_d, q) -> np.ndarray:
    """6x6 base-frame geometric Jacobian (top 3 rows linear, bottom 3 angular).

    For each revolute joint ``i`` the rotation ``Rz(theta_i)`` acts about the z-axis of the
    *pre-joint* frame ``T_pre_i = (link_0..link_{i-1}) · Rx(alpha_i)·Tx(a_i)·Tz(d_i)`` (i.e.
    after ``Rx·Tx·Tz`` but before ``Rz``). Using ``T_pre_i``'s z-axis and origin is what
    reproduces the textbook singularities; the naive "z of the full frame i" placement does
    not.
    """
    n = len(q)
    pre_frames: list[np.ndarray] = []
    cum = np.eye(4, dtype=np.float64)
    for i in range(n):
        pre = cum @ _rot_x(dh_alpha[i]) @ _trans_x(dh_a[i]) @ _trans_z(dh_d[i])
        pre_frames.append(pre)
        cum = pre @ _rot_z(q[i])  # full link_i applied -> base->frame{i+1}
    o_n = cum[:3, 3]

    jac = np.zeros((6, n), dtype=np.float64)
    for i in range(n):
        z_i = pre_frames[i][:3, 2]
        o_i = pre_frames[i][:3, 3]
        jac[:3, i] = np.cross(z_i, o_n - o_i)
        jac[3:, i] = z_i
    return jac


def manipulability(dh_alpha, dh_a, dh_d, q) -> float:
    """Yoshikawa manipulability ``w = |det(J)|`` (tool- and frame-invariant; ``→0`` at any singularity)."""
    return float(abs(np.linalg.det(geometric_jacobian(dh_alpha, dh_a, dh_d, q))))


def damping_scale(w: float, w_low: float, w_high: float, s_min: float) -> float:
    """Map manipulability ``w`` to a command scale ``s`` in ``[s_min, 1]``.

    ``s = clamp((w - w_low) / (w_high - w_low), s_min, 1)`` — full speed (``1``) when
    well-conditioned (``w >= w_high``), floored at ``s_min`` (``> 0`` so the operator can
    always slowly escape) when near-singular (``w <= w_low``), linearly ramped between.
    """
    if w_high <= w_low:
        return 1.0
    s = (w - w_low) / (w_high - w_low)
    return float(min(max(s, s_min), 1.0))


def directional_scale(
    dh_alpha,
    dh_a,
    dh_d,
    q,
    dx: np.ndarray,
    w_low: float,
    w_high: float,
    s_min: float,
    lam: float = 1e-3,
    eps: float = 1e-3,
) -> tuple[float, float]:
    """Direction-aware damping: don't resist a step that *escapes* the singularity.

    Returns ``(s, w)``. Computes the damped-least-squares joint step ``dq`` for the desired
    Cartesian step ``dx`` (6-vec: linear + angular), then re-evaluates ``w`` a tiny step
    along ``dq``. If the move increases manipulability (escaping), returns ``s=1`` (no
    damping); otherwise returns the scalar ``damping_scale``. ``lam`` damps the pseudo-inverse
    so it stays finite at the singularity.
    """
    jac = geometric_jacobian(dh_alpha, dh_a, dh_d, q)
    w = float(abs(np.linalg.det(jac)))
    s = damping_scale(w, w_low, w_high, s_min)
    if s >= 1.0:
        return 1.0, w
    jjt = jac @ jac.T
    dq = jac.T @ np.linalg.solve(jjt + (lam * lam) * np.eye(6), np.asarray(dx, dtype=np.float64))
    nrm = float(np.linalg.norm(dq))
    if nrm > 1e-12:
        q2 = np.asarray(q, dtype=np.float64) + eps * dq / nrm
        w2 = float(abs(np.linalg.det(geometric_jacobian(dh_alpha, dh_a, dh_d, q2))))
        if w2 >= w:
            return 1.0, w
    return s, w


def joint_velocity_scale(
    dh_alpha,
    dh_a,
    dh_d,
    q,
    dx: np.ndarray,
    dt: float,
    qdot_limit: np.ndarray,
    lam: float = 1e-4,
) -> tuple[float, np.ndarray]:
    """Uniform Cartesian-step scale keeping the *predicted* joint step under per-joint velocity limits.

    Predicts the joint step for the desired Cartesian step ``dx`` (6-vec: linear + angular, in the
    arm base frame) via the damped-least-squares pseudo-inverse of the geometric Jacobian::

        dq = J^T (J J^T + lam^2 I)^-1 dx

    and returns ``(s, dq)`` with ``s`` in ``(0, 1]`` such that scaling ``dx`` by ``s`` keeps every
    joint under its per-tick budget::

        ratio = max_i |dq_i| / (qdot_limit_i * dt)
        s     = 1 / ratio   if ratio > 1   else   1.0

    Scaling the whole Cartesian step by one scalar preserves the *direction* of end-effector motion
    (the MoveIt Servo joint-limit principle) rather than distorting it by clamping joints
    independently. This is the host-side analogue of the Elite controller's per-joint
    ``JOINT_IGNORE_SPEED`` check: it slows/holds a command *before* the controller's internal IK
    spikes joint velocity past that bound and drops external control.

    The guard is **naturally directional**: near a singularity a step that drives *into* the
    singular set demands a huge ``dq`` -> small ``s`` (held), while an escaping step demands a
    moderate ``dq`` -> ``s ≈ 1`` (allowed). No ``s_min`` floor is applied — a floor would let a
    limit-exceeding command through; a fully-held offending direction is the correct, safe outcome
    and escape stays possible because escape directions are not scaled.

    ``lam`` is a **predictor-fidelity** knob, NOT a robustness smoother, and this distinction is
    load-bearing. This ``dq`` must model what the *controller's* internal IK will actually command
    (which is near-exact / lightly-damped and spikes as ``1/sigma_min`` near a singularity), so that
    we hold BEFORE the controller trips. The DLS gain on a singular direction is
    ``sigma / (sigma^2 + lam^2)``, which tracks the exact ``1/sigma`` only while ``lam << sigma`` and
    otherwise CAPS the predicted velocity at ``1/(2 lam)`` — i.e. a ``lam`` on the order of the
    operating ``sigma_min`` (~1e-2) silently *under*-predicts the very spike we must catch (verified:
    with ``lam=1e-2`` the guard-approved step still drove the true joint velocity to >7x its budget at
    a wrist singularity). Keep ``lam`` a couple orders of magnitude below the smallest ``sigma`` you
    care to protect (default 1e-4 vs the ~4e-3 ``sigma_min`` seen entering the trip); it only needs to
    stay large enough to keep the solve finite. At the *exact* singularity the DLS gain of that
    direction is 0, so ``dq -> 0`` and ``s = 1`` — harmless, since the unreachable direction also
    carries no motion. ``qdot_limit`` is the per-joint velocity ceiling (rad/s, length-6) already
    scaled by any headroom margin. A zero ``dx`` (no motion) yields ``dq = 0`` and ``s = 1``.
    """
    jac = geometric_jacobian(dh_alpha, dh_a, dh_d, q)
    dx = np.asarray(dx, dtype=np.float64)
    jjt = jac @ jac.T
    dq = jac.T @ np.linalg.solve(jjt + (lam * lam) * np.eye(6), dx)
    qdot_limit = np.asarray(qdot_limit, dtype=np.float64)
    budget = qdot_limit * dt
    # Guard against a non-positive budget (bad config) — treat as no scaling rather than div-by-zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(budget > 0.0, np.abs(dq) / budget, 0.0)
    ratio = float(np.max(ratios)) if ratios.size else 0.0
    s = 1.0 / ratio if ratio > 1.0 else 1.0
    return s, dq
