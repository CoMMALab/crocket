"""The algorithms, ported from RBDReference (which stays the oracle), written once over the
traced spatial types. Joint loops unroll at trace time; mimic joints feed their mimicked column
scaled by their multiplier, as in RBDReference.

Outputs: poses are ``[t (3), R column-major (9)]`` per frame; the Jacobian is row-major
``6 x n_q``, rows ``(vx, vy, vz, wx, wy, wz)``, LOCAL_WORLD_ALIGNED at the frame origin;
matrices are ``n_q x n_q`` column-major.
"""

from __future__ import annotations

import math

from .trace import Motion, SE3, Scalar, algorithm


def _poses(robot, q):
    """World-to-joint transform of every joint."""
    X0 = {}
    for j in robot.joints:
        X = robot.Xup(j, robot.q(j, q))
        p = robot.parent(j)
        X0[j] = X if p < 0 else X @ X0[p]
    return X0


def _frame(robot, X0, name):
    f = robot.frame(name)
    Xf = SE3.const_pose(robot.t, f.T)
    return Xf if f.parent < 0 else Xf @ X0[f.parent]


def _pose_out(X):
    """``[t, R column-major]`` of a world-to-frame transform (R = E^T, so R's columns are E's rows)."""
    e = X.elems
    return [e[9:12], e[:9]]


@algorithm(inputs=("q",), outputs=("poses",))
def frame_poses(robot, q, frames=()):
    X0 = _poses(robot, q)
    return [_pose_out(_frame(robot, X0, name)) for name in frames]


def _jacobian(robot, X0, frame):
    """World transform of ``frame`` and its LOCAL_WORLD_ALIGNED Jacobian, rows (v, w)."""
    Xe = _frame(robot, X0, frame)
    p = Xe.translation
    f = robot.frame(frame)
    zero = Scalar.const(robot.t, 0.0)
    J = [[zero] * robot.n_q for _ in range(6)]
    for j in (robot.ancestors(f.parent) if f.parent >= 0 else []):
        col = (X0[j].inv() @ robot.S(j)).at(p)  # world-aligned twist at the frame origin
        for row, k in enumerate((3, 4, 5, 0, 1, 2)):
            J[row][robot.col(j)] = J[row][robot.col(j)] + col[k] * robot.scale(j)
    return Xe, J


@algorithm(inputs=("q",), outputs=("pose", "jacobian"))
def frame_jacobian(robot, q, frame=None):
    Xe, J = _jacobian(robot, _poses(robot, q), frame)
    return _pose_out(Xe), J


def _so3_log(R, eps=1e-12):
    """``log(R)^vee``, smooth through theta = 0: the Taylor form ``vee / 2`` blended with the
    exact ``theta / (2 sin theta) vee`` by ``w = sin^2 / (sin^2 + eps)`` (McVAMP's so3_log_smooth).
    ``sin theta = |vee| / 2`` and ``theta = atan2(sin, cos)`` rather than McVAMP's ``acos`` and
    ``1 - cos^2``: those cancel near pi and leave float32 three digits there."""
    vee = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]
    s2 = (vee[0] * vee[0] + vee[1] * vee[1] + vee[2] * vee[2]) * 0.25
    s = s2.sqrt()
    c = (R[0][0] + R[1][1] + R[2][2] - 1.0) * 0.5
    w = s2 / (s2 + eps)
    k = (1.0 - w) * 0.5 + w * s.atan2(c) / (2.0 * s + eps)
    return [v * k for v in vee]


def _cholesky_solve(A, b):
    """``A^-1 b`` for symmetric positive definite ``A``, unrolled (fixed size, branch-free)."""
    n = len(b)
    L = [[None] * n for _ in range(n)]
    for i in range(n):
        for k in range(i + 1):
            acc = A[i][k]
            for j in range(k):
                acc = acc - L[i][j] * L[k][j]
            L[i][k] = acc.sqrt() if i == k else acc / L[k][k]
    y = [None] * n
    for i in range(n):
        acc = b[i]
        for j in range(i):
            acc = acc - L[i][j] * y[j]
        y[i] = acc / L[i][i]
    x = [None] * n
    for i in reversed(range(n)):
        acc = y[i]
        for j in range(i + 1, n):
            acc = acc - L[j][i] * x[j]
        x[i] = acc / L[i][i]
    return x


@algorithm(inputs=("q", ("target", lambda o: 7), ("lam", lambda o: 1)),
           outputs=("q_next", "cost", "pos_err2", "ori_err2"))
def ik_lm_step(robot, q, target, lam, frame=None, pos_weight=50.0, ori_weight=10.0):
    """One damped least-squares step of single-frame pose IK, projected onto the joint limits.

    ``target`` is the frame's goal pose ``[qw, qx, qy, qz, x, y, z]``; ``lam`` the damping. The
    residual is ``r = [w_p (p - p*); w_o log(R R*^T)]`` with the world-frame Jacobian (the
    orientation rows are the angular-velocity rows, log's first-order Jacobian). The step is the
    inner Levenberg-Marquardt form ``J^T (J J^T + lam I)^-1 r`` (6x6 whatever the robot), then
    ``q - step`` is clamped to the joint limits. ``cost = |r|^2`` is at the input ``q``. The
    loop (acceptance, damping updates, convergence) belongs to the caller.
    """
    Xe, J = _jacobian(robot, _poses(robot, q), frame)
    R = [[Xe[3 * k + i] for k in range(3)] for i in range(3)]  # world rotation: E^T
    qw, qx, qy, qz = target[0], target[1], target[2], target[3]
    G = [[1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
         [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
         [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]]
    Rerr = [[R[i][0] * G[j][0] + R[i][1] * G[j][1] + R[i][2] * G[j][2] for j in range(3)] for i in range(3)]
    ep = [Xe[9 + i] - target[4 + i] for i in range(3)]
    eo = _so3_log(Rerr)
    r = [e * pos_weight for e in ep] + [e * ori_weight for e in eo]
    Jw = [[x * (pos_weight if row < 3 else ori_weight) for x in J[row]] for row in range(6)]
    A = [[sum((Jw[i][c] * Jw[k][c] for c in range(robot.n_q)), lam[0] if i == k else Scalar.const(robot.t, 0.0))
          for k in range(6)] for i in range(6)]
    z = _cholesky_solve(A, r)
    q_next = []
    for c in range(robot.n_q):
        step = sum((Jw[row][c] * z[row] for row in range(6)), Scalar.const(robot.t, 0.0))
        jt = next(j for j in robot.robot.joints if j.q_index == c and not j.mimic)
        lo, hi = max(jt.limits[0], -1e30), min(jt.limits[1], 1e30)
        q_next.append((q[c] - step).maximum(lo).minimum(hi))
    sq = lambda v: sum((x * x for x in v), Scalar.const(robot.t, 0.0))  # noqa: E731
    return q_next, sq(r), sq(ep), sq(eo)


def _pose_residual(robot, q, target, frame):
    """Jacobian and ``(p - p*, log(R R*^T))`` of ``frame`` at ``q``: GRiM's pose residual."""
    Xe, J = _jacobian(robot, _poses(robot, q), frame)
    R = [[Xe[3 * k + i] for k in range(3)] for i in range(3)]
    qw, qx, qy, qz = target[0], target[1], target[2], target[3]
    G = [[1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
         [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
         [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]]
    Rerr = [[R[i][0] * G[j][0] + R[i][1] * G[j][1] + R[i][2] * G[j][2] for j in range(3)] for i in range(3)]
    return J, [Xe[9 + i] - target[4 + i] for i in range(3)], _so3_log(Rerr)


@algorithm(inputs=("q", ("target", lambda o: 7)), outputs=("cost",))
def ik_pose_cost(robot, q, target, frame=None, pos_weight=50.0, ori_weight=10.0):
    """``|W r|^2`` at ``q``: the merit of GRiM ``ls_ik``'s line search (the Jacobian is dead
    code here and drops out of the trace)."""
    _, ep, eo = _pose_residual(robot, q, target, frame)
    w = [pos_weight] * 3 + [ori_weight] * 3
    return sum((e * wk * (e * wk) for e, wk in zip(ep + eo, w)), Scalar.const(robot.t, 0.0))


@algorithm(inputs=("q", ("target", lambda o: 7), ("lam", lambda o: 1)),
           outputs=("delta", "cost", "pos_err2", "ori_err2"))
def ik_ls_step(robot, q, target, lam, frame=None, pos_weight=50.0, ori_weight=10.0):
    """GRiM ``ls_ik``'s step direction (kernels/ik/ls_ik.cu): weighted residual and Jacobian,
    Jacobi column scaling, the outer normal equations ``(Js^T Js + lam I) d = -Js^T W r`` over
    the frame's chain joints (n x n, where ik_lm_step solves 6 x 6), unscaled. ``delta`` is zero
    off the chain; ``cost = |W r|^2`` at ``q``. Trust region, line search and acceptance belong
    to the caller, as in GRiM."""
    cols = ik_chain_cols(robot, frame)
    Jw, r, cost, pe2, oe2 = _ls_system(robot, q, target, frame, pos_weight, ori_weight, cols)
    zero = Scalar.const(robot.t, 0.0)
    delta = [zero] * robot.n_q
    for c, d in zip(cols, _ls_solve(robot, [x for row in Jw for x in row], r, lam, len(cols))):
        delta[c] = d
    return delta, cost, pe2, oe2


def ik_chain_cols(robot, frame):
    """q columns that move ``frame``: the unknowns of GRiM's chain-only solve."""
    f = robot.frame(frame)
    return sorted({robot.col(j) for j in (robot.ancestors(f.parent) if f.parent >= 0 else [])})


def _ls_system(robot, q, target, frame, pos_weight, ori_weight, cols):
    J, ep, eo = _pose_residual(robot, q, target, frame)
    zero = Scalar.const(robot.t, 0.0)
    w = [pos_weight] * 3 + [ori_weight] * 3
    r = [e * wk for e, wk in zip(ep + eo, w)]
    sq = lambda v: sum((x * x for x in v), zero)  # noqa: E731
    return [[J[k][c] * w[k] for c in cols] for k in range(6)], r, sq(r), sq(ep), sq(eo)


def _ls_solve(robot, Jw, r, lam, n):
    """Jacobi-scaled ``(Js^T Js + lam I) d = -Js^T r``, unscaled; ``Jw`` row-major 6 x n."""
    zero = Scalar.const(robot.t, 0.0)
    Jw = [Jw[n * k:n * k + n] for k in range(6)]
    scale = [sum((Jw[k][i] * Jw[k][i] for k in range(6)), zero).sqrt() + 1e-8 for i in range(n)]
    Js = [[Jw[k][i] / scale[i] for i in range(n)] for k in range(6)]
    A = [[sum((Js[k][i] * Js[k][j] for k in range(6)), lam[0] if i == j else zero) for j in range(n)]
         for i in range(n)]
    d = _cholesky_solve(A, [-sum((Js[k][i] * r[k] for k in range(6)), zero) for i in range(n)])
    return [di / s for di, s in zip(d, scale)]


@algorithm(inputs=("q", ("target", lambda o: 7)), outputs=("Jw", "r", "cost", "pos_err2", "ori_err2"))
def ik_ls_system(robot, q, target, frame=None, pos_weight=50.0, ori_weight=10.0):
    """First half of ``ik_ls_step``: the weighted chain Jacobian (row-major 6 x n_chain), weighted
    residual and errors. With ``ik_ls_solve`` it lets the two halves run at different precisions,
    as GRiM's ls_ik does (float kinematics, double normal equations)."""
    return _ls_system(robot, q, target, frame, pos_weight, ori_weight, ik_chain_cols(robot, frame))


@algorithm(inputs=(("Jw", lambda o: 6 * o["n"]), ("r", lambda o: 6), ("lam", lambda o: 1)),
           outputs=("delta",))
def ik_ls_solve(robot, Jw, r, lam, n=0):
    """Second half of ``ik_ls_step``: the step over the ``n`` chain columns."""
    return _ls_solve(robot, Jw, r, lam, n)


@algorithm(inputs=(("Jw", lambda o: 6 * o["n"]), ("r", lambda o: 6), ("lam", lambda o: 1)),
           outputs=("H", "g", "scale"))
def ik_sqp_system(robot, Jw, r, lam, n=0):
    """GRiM ``sqp_ik``'s QP in Jacobi-scaled space (kernels/ik/sqp_ik.cu): ``H = Js^T Js + lam I``
    (row-major n x n), ``g = Js^T r`` and the column scales, ``Js = Jw / scale``. The active-set
    loop over the box bounds belongs to the caller; it re-solves with ``ik_masked_solve``."""
    zero = Scalar.const(robot.t, 0.0)
    Jw = [Jw[n * k:n * k + n] for k in range(6)]
    scale = [sum((Jw[k][i] * Jw[k][i] for k in range(6)), zero).sqrt() + 1e-8 for i in range(n)]
    Js = [[Jw[k][i] / scale[i] for i in range(n)] for k in range(6)]
    H = [sum((Js[k][i] * Js[k][j] for k in range(6)), lam[0] if i == j else zero)
         for i in range(n) for j in range(n)]
    return H, [sum((Js[k][i] * r[k] for k in range(6)), zero) for i in range(n)], scale


@algorithm(inputs=(("H", lambda o: o["n"] ** 2), ("g", lambda o: o["n"]), ("pb", lambda o: o["n"]),
                   ("active", lambda o: o["n"])), outputs=("p",))
def ik_masked_solve(robot, H, g, pb, active, n=0):
    """One active-set re-solve of GRiM ``sqp_ik``, branch-free: ``active`` is 1 for a joint held
    at its bound ``pb`` (0 for free joints, whose ``pb`` must be 0). Free joints solve
    ``H_ff p_f = -(g + H pb)_f``; active rows are identity with right-hand side 0, and the caller
    puts ``pb`` back. With ``active = 0`` it is the unconstrained Newton step."""
    one = Scalar.const(robot.t, 1.0)
    H = [H[n * i:n * i + n] for i in range(n)]
    free = [one - a for a in active]
    A = [[H[i][j] * (free[i] * free[j]) + (active[i] if i == j else 0.0) for j in range(n)]
         for i in range(n)]
    rhs = [-(g[i] + sum((H[i][j] * pb[j] for j in range(n)), Scalar.const(robot.t, 0.0))) * free[i]
           for i in range(n)]
    return (_cholesky_solve(A, rhs),)


def _sphere_centres(robot, X0, spheres):
    """World centres ``t + E^T c`` of ``(frame, centre)`` spheres, flat ``3 * len(spheres)``."""
    out = []
    for name, centre in spheres:
        X = _frame(robot, X0, name)
        p = [X[9 + i] for i in range(3)]
        for k in range(3):
            if centre[k] != 0.0:
                for i in range(3):
                    p[i] = p[i] + X[3 * k + i] * float(centre[k])
        out += p
    return out


def _ik_geometry(robot, q, targets, frames, cols, wspheres, sspheres, jacobian):
    X0 = _poses(robot, q)
    zero = Scalar.const(robot.t, 0.0)
    r, J = [], []
    for e, name in enumerate(frames):
        Xe, Jf = _jacobian(robot, X0, name)
        R = [[Xe[3 * k + i] for k in range(3)] for i in range(3)]
        qw, qx, qy, qz = (targets[7 * e + i] for i in range(4))
        G = [[1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
             [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
             [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]]
        Rerr = [[R[i][0] * G[j][0] + R[i][1] * G[j][1] + R[i][2] * G[j][2] for j in range(3)]
                for i in range(3)]
        r += [Xe[9 + i] - targets[7 * e + 4 + i] for i in range(3)] + _so3_log(Rerr)
        J += [Jf[row][c] for row in range(6) for c in cols]
    cw, cs = _sphere_centres(robot, X0, wspheres), _sphere_centres(robot, X0, sspheres)
    if not jacobian:
        return r, cw, cs
    tw = []
    for j in robot.joints:  # world twist at the world origin, [w, v_o], times the mimic scale
        m = X0[j].inv() @ robot.S(j)
        tw += [m[k] * robot.scale(j) for k in range(6)]
    return r, J, cw, cs, (tw if (wspheres or sspheres) else [zero] * 0)


@algorithm(inputs=("q", ("targets", lambda o: 7 * len(o.get("frames", ())))),
           outputs=("r", "J", "cw", "cs", "tw"))
def ik_geometry(robot, q, targets, frames=(), cols=(), wspheres=(), sspheres=()):
    """Everything an IK iteration reads from the robot, for several target frames at once.

    ``r`` is the stacked unweighted pose residual ``[p - p*, log(R R*^T)]`` per frame (6 each)
    against ``targets`` (``[qw, qx, qy, qz, x, y, z]`` per frame), ``J`` the stacked
    LOCAL_WORLD_ALIGNED Jacobian over the q columns ``cols`` (row-major ``6 n_frames x
    len(cols)``). ``cw`` / ``cs`` are the world centres of the ``(frame, centre)`` spheres
    ``wspheres`` (world collision) and ``sspheres`` (self collision), 3 each. ``tw`` is every
    joint's world twist at the world origin ``[w, v_o]`` times its mimic scale, 6 per joint in
    ``robot.joints`` order (empty without spheres): a sphere centre ``c`` moves with
    ``v_o + w x c`` per unit of that joint's column, which is all a collision gradient needs.
    Distances to runtime obstacles and the active set belong to the caller."""
    return _ik_geometry(robot, q, targets, frames, cols, wspheres, sspheres, True)


@algorithm(inputs=("q", ("targets", lambda o: 7 * len(o.get("frames", ())))),
           outputs=("r", "cw", "cs"))
def ik_geometry_value(robot, q, targets, frames=(), cols=(), wspheres=(), sspheres=()):
    """``ik_geometry`` without the Jacobian and twists: the merit of a line-search trial."""
    return _ik_geometry(robot, q, targets, frames, cols, wspheres, sspheres, False)


def _min_tree(xs):
    """Balanced min reduction: depth log2(n), so the cooperative tiers can split it."""
    while len(xs) > 1:
        xs = [a.minimum(b) for a, b in zip(xs[::2], xs[1::2])] + ([xs[-1]] if len(xs) % 2 else [])
    return xs[0]


@algorithm(inputs=("q", ("boxes", lambda o: 15 * o.get("n_boxes", 0))), outputs=("margin",))
def sphere_collision(robot, q, boxes, spheres=(), n_boxes=0):
    """Fused FK and environment collision: ``min(d^2 - r^2)`` over every robot sphere and box,
    ``d`` the distance from the sphere's centre to the box; negative means in collision.

    ``spheres`` are ``(frame or link name, centre in that frame (3), radius)``, folded in as
    constants. ``boxes`` is ``n_boxes`` rows of ``[centre (3), unit axes (3 x 3), half-extents (3)]``,
    per call.
    """
    X0 = _poses(robot, q)
    terms = []
    for name, centre, radius in spheres:
        X = _frame(robot, X0, name)  # elements [E row-major (9), t (3)]; world point t + E^T c
        p = [X[9 + i] for i in range(3)]
        for k in range(3):
            if centre[k] != 0.0:
                for i in range(3):
                    p[i] = p[i] + X[3 * k + i] * float(centre[k])
        r2 = float(radius) ** 2
        for b in range(n_boxes):
            box = boxes[15 * b:15 * b + 15]
            rel = [p[i] - box[i] for i in range(3)]
            d2 = Scalar.const(robot.t, 0.0)
            for k in range(3):
                axis = box[3 + 3 * k:6 + 3 * k]
                x = (rel[0] * axis[0] + rel[1] * axis[1] + rel[2] * axis[2]).abs() - box[12 + k]
                x = x.maximum(0.0)
                d2 = d2 + x * x
            terms.append(d2 - r2)
    return _min_tree(terms) if terms else Scalar.const(robot.t, float("inf"))


@algorithm(inputs=("q", "qd", "qdd"), outputs=("tau",))
def rnea(robot, q, qd, qdd):
    v, a, f = {}, {}, {}
    X = {}
    for j in robot.joints:
        X[j] = robot.Xup(j, robot.q(j, q))
        p = robot.parent(j)
        S, k = robot.S(j), robot.scale(j)
        vJ = S * (qd[robot.col(j)] * k)
        v[j] = vJ if p < 0 else X[j] @ v[p] + vJ
        a[j] = X[j] @ (robot.gravity if p < 0 else a[p]) + S * (qdd[robot.col(j)] * k) + v[j].cross(vJ)
        I = robot.I(j)
        f[j] = I @ a[j] + v[j].cross_dual(I @ v[j])
    tau = [Scalar.const(robot.t, 0.0)] * robot.n_q
    for j in reversed(robot.joints):
        c = robot.col(j)
        tau[c] = tau[c] + robot.S(j).dot(f[j]) * robot.scale(j)
        p = robot.parent(j)
        if p >= 0:
            f[p] = f[p] + X[j].T @ f[j]
    return tau


@algorithm(inputs=("q",), outputs=("M",))
def crba(robot, q):
    n = robot.n_q
    X = {j: robot.Xup(j, robot.q(j, q)) for j in robot.joints}
    Ic = {j: robot.I(j) for j in robot.joints}
    zero = Scalar.const(robot.t, 0.0)
    M = [[zero] * n for _ in range(n)]
    for j in reversed(robot.joints):
        p = robot.parent(j)
        if p >= 0:
            Ic[p] = Ic[p] + X[j].T @ Ic[j] @ X[j]
        F = Ic[j] @ robot.S(j)
        i, cj, kj = j, robot.col(j), robot.scale(j)
        while i >= 0:
            ci, entry = robot.col(i), robot.S(i).dot(F) * (kj * robot.scale(i))
            M[ci][cj] = M[ci][cj] + entry
            if i != j:
                M[cj][ci] = M[cj][ci] + entry
            if robot.parent(i) >= 0:
                F = X[i].T @ F
            i = robot.parent(i)
    return [M[r][c] for c in range(n) for r in range(n)]


@algorithm(inputs=("q", "qd", "tau"), outputs=("qdd",))
def aba(robot, q, qd, tau):
    if any(jt.mimic for jt in robot.robot.joints):
        raise ValueError("ABA does not support mimic joints; use CRBA and a Cholesky solve")
    X, v, c, IA, pA = {}, {}, {}, {}, {}
    for j in robot.joints:
        X[j] = robot.Xup(j, robot.q(j, q))
        p = robot.parent(j)
        vJ = robot.S(j) * qd[robot.col(j)]
        v[j] = vJ if p < 0 else X[j] @ v[p] + vJ
        c[j] = v[j].cross(vJ)
        IA[j] = robot.I(j)
        pA[j] = v[j].cross_dual(IA[j] @ v[j])
    U, dinv, u = {}, {}, {}
    for j in reversed(robot.joints):
        S = robot.S(j)
        U[j] = IA[j] @ S
        dinv[j] = 1.0 / S.dot(U[j])
        u[j] = tau[robot.col(j)] - S.dot(pA[j])
        p = robot.parent(j)
        if p >= 0:
            Ia = IA[j] - U[j].outer(U[j] * dinv[j])
            pa = pA[j] + Ia @ c[j] + U[j] * (u[j] * dinv[j])
            IA[p] = IA[p] + X[j].T @ Ia @ X[j]
            pA[p] = pA[p] + X[j].T @ pa
    a, qdd = {}, [None] * robot.n_q
    for j in robot.joints:
        p = robot.parent(j)
        a[j] = X[j] @ (robot.gravity if p < 0 else a[p]) + c[j]
        qdd[robot.col(j)] = (u[j] - U[j].dot(a[j])) * dinv[j]
        a[j] = a[j] + robot.S(j) * qdd[robot.col(j)]
    return qdd


@algorithm(inputs=("q", "qd", "qdd"), outputs=("dtau_dq", "dtau_dqd"))
def rnea_derivatives(robot, q, qd, qdd):
    """Analytical RNEA derivatives (forward-mode, one tangent per column of q and of qd).

    ``dX_j/dq_j y = (X_j y) x S_j``, so a joint's own coordinate enters through a cross
    product with its subspace; tangents that are structurally zero are skipped (None).
    """
    n = robot.n_q
    X, v, a, f, I = {}, {}, {}, {}, {}
    for j in robot.joints:
        X[j] = robot.Xup(j, robot.q(j, q))
        p = robot.parent(j)
        S, k = robot.S(j), robot.scale(j)
        vJ = S * (qd[robot.col(j)] * k)
        v[j] = vJ if p < 0 else X[j] @ v[p] + vJ
        a[j] = X[j] @ (robot.gravity if p < 0 else a[p]) + S * (qdd[robot.col(j)] * k) + v[j].cross(vJ)
        I[j] = robot.I(j)
        f[j] = I[j] @ a[j] + v[j].cross_dual(I[j] @ v[j])

    def add(x, y):
        return y if x is None else x if y is None else x + y

    out = []
    for wrt in ("q", "qd"):
        dv, da, df = {}, {}, {}
        for j in robot.joints:
            p = robot.parent(j)
            S, k, cj = robot.S(j), robot.scale(j), robot.col(j)
            vJ = S * (qd[cj] * k)
            Xvp = None if p < 0 else X[j] @ v[p]
            Xap = X[j] @ (robot.gravity if p < 0 else a[p])
            for col in range(n):
                dvj = None if p < 0 or (p, col) not in dv else X[j] @ dv[p, col]
                daj = None if p < 0 or (p, col) not in da else X[j] @ da[p, col]
                if col == cj:
                    if wrt == "q":
                        if Xvp is not None:
                            dvj = add(dvj, Xvp.cross(S) * k)
                        daj = add(daj, Xap.cross(S) * k)
                    else:
                        dvj = add(dvj, S * k)
                        daj = add(daj, v[j].cross(S * k))
                if dvj is not None:
                    daj = add(daj, dvj.cross(vJ))
                if dvj is None and daj is None:
                    continue
                if dvj is not None:
                    dv[j, col] = dvj
                da[j, col] = daj
                dfj = None if daj is None else I[j] @ daj
                if dvj is not None:
                    dfj = add(dfj, dvj.cross_dual(I[j] @ v[j]) + v[j].cross_dual(I[j] @ dvj))
                df[j, col] = dfj
        zero = Scalar.const(robot.t, 0.0)
        D = [[zero] * n for _ in range(n)]
        for j in reversed(robot.joints):
            p, S, k, cj = robot.parent(j), robot.S(j), robot.scale(j), robot.col(j)
            for col in range(n):
                if (j, col) in df:
                    D[cj][col] = D[cj][col] + S.dot(df[j, col]) * k
                if p < 0:
                    continue
                dfp = None if (j, col) not in df else X[j].T @ df[j, col]
                if wrt == "q" and col == cj:
                    dfp = add(dfp, X[j].T @ S.cross_dual(f[j]) * k)
                if dfp is not None:
                    df[p, col] = add(df.get((p, col)), dfp)
            if p >= 0:
                f[p] = f[p] + X[j].T @ f[j]
        out.append([D[r][c] for c in range(n) for r in range(n)])
        # restore the forward-pass forces for the qd pass
        for j in robot.joints:
            f[j] = I[j] @ a[j] + v[j].cross_dual(I[j] @ v[j])
    return tuple(out)


# ---------------------------------------------------------------------------
# Closed-form IK of the 7-DOF "spherical shoulder, intersecting axes 5-6, offset axis 7" family
# (Franka Panda / FR3): GRiM's kernels/ik/analytic_ik.cu, traced with its arm geometry folded in.
# ---------------------------------------------------------------------------

_SP_TOL, _TINY = 1e-6, 1e-12


class _Arm:
    """The analytic solver's geometry (GRiM's ArmGeom field order, 95 scalars) as traced
    constants, plus the scalar helpers the closed form needs."""

    def __init__(self, t, geom):
        g = [float(x) for x in geom]
        assert len(g) == 95, len(g)
        c = lambda v: Scalar.const(t, v)  # noqa: E731
        self.t, self.c = t, c
        self.axes = [[c(v) for v in g[3 * i:3 * i + 3]] for i in range(7)]
        self.points = [[c(v) for v in g[21 + 3 * i:24 + 3 * i]] for i in range(7)]
        self.axes_f = [g[3 * i:3 * i + 3] for i in range(7)]
        self.shoulder = [c(v) for v in g[42:45]]
        self.wrist = [c(v) for v in g[45:48]]
        self.m_home = [c(v) for v in g[48:64]]
        self.m_home_inv = [c(v) for v in g[64:80]]
        self.cos_alpha = c(g[80])

    # -- small linear algebra over traced scalars -------------------------------------------
    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    def norm(self, a):
        return self.dot(a, a).sqrt()

    def unit(self, a):
        n = self.norm(a).maximum(_TINY)   # the kernel's `n > TINY ? n : 1`, smooth in n
        return [x / n for x in a]

    @staticmethod
    def mv(M, v):
        return [M[3 * i] * v[0] + M[3 * i + 1] * v[1] + M[3 * i + 2] * v[2] for i in range(3)]

    @staticmethod
    def mm(A, B):
        return [A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]
                for i in range(3) for j in range(3)]

    @staticmethod
    def mtm(A, B):
        return [A[i] * B[j] + A[3 + i] * B[3 + j] + A[6 + i] * B[6 + j] for i in range(3) for j in range(3)]

    @staticmethod
    def m4(A, B):
        return [sum((A[4 * i + k] * B[4 * k + j] for k in range(1, 4)), A[4 * i] * B[j])
                for i in range(4) for j in range(4)]

    def rot(self, k, th):
        cth, sth = th.cos(), th.sin()
        v = 1.0 - cth
        x, y, z = k
        return [cth + x * x * v, x * y * v - z * sth, x * z * v + y * sth,
                x * y * v + z * sth, cth + y * y * v, y * z * v - x * sth,
                x * z * v - y * sth, y * z * v + x * sth, cth + z * z * v]

    @staticmethod
    def wrap(th):
        return th.sin().atan2(th.cos())

    def acos(self, x):
        return (1.0 - x * x).maximum(0.0).sqrt().atan2(x)

    def screw(self, i, th):
        R = self.rot(self.axes[i], th)
        p = self.points[i]
        Rp = self.mv(R, p)
        z, o = self.c(0.0), self.c(1.0)
        return [R[0], R[1], R[2], p[0] - Rp[0], R[3], R[4], R[5], p[1] - Rp[1],
                R[6], R[7], R[8], p[2] - Rp[2], z, z, z, o]

    # -- Paden-Kahan subproblems -------------------------------------------------------------
    def sp4(self, h, p, k, d):
        """h . rot(k, th) p = d -> two roots. Past the sinusoid's reach the clamped ratio gives
        acos = 0 or pi: the residual-minimising angle, as the kernel's least-squares branch."""
        kp, hk = self.dot(k, p), self.dot(h, k)
        A = self.dot(h, p) - hk * kp
        B = self.dot(h, self.cross(k, p))
        R = (A * A + B * B).sqrt()
        phi = B.atan2(A)
        ratio = ((d - hk * kp) / R.maximum(_SP_TOL)).maximum(-1.0).minimum(1.0)
        delta = self.acos(ratio)
        return self.wrap(phi + delta), self.wrap(phi - delta)

    def sp3(self, p1, p2, k, d):
        return self.sp4(p2, p1, k, 0.5 * (self.dot(p1, p1) + self.dot(p2, p2) - d * d))

    def sp1(self, p1, p2, k):
        kp1, kp2 = self.dot(k, p1), self.dot(k, p2)
        a = [p1[i] - k[i] * kp1 for i in range(3)]
        b = [p2[i] - k[i] * kp2 for i in range(3)]
        return self.wrap(self.dot(k, self.cross(a, b)).atan2(self.dot(a, b)))

    def sp2(self, p1, p2, k1, k2):
        t2 = self.sp4(k1, p2, k2, self.dot(k1, p1))
        t1 = tuple(self.sp1(p1, self.mv(self.rot(k2, t), p2), k1) for t in t2)
        return t1, t2

    def rot_two(self, a1, a2, b1, b2):
        """The rotation carrying a1 -> b1 and a2 -> b2 (Gram-Schmidt frames)."""
        def frame(u, v):
            e1 = self.unit(u)
            d = self.dot(e1, v)
            e2 = self.unit([v[i] - e1[i] * d for i in range(3)])
            e3 = self.cross(e1, e2)
            return [e1[0], e2[0], e3[0], e1[1], e2[1], e3[1], e1[2], e2[2], e3[2]]
        E, F = frame(a1, a2), frame(b1, b2)
        Et = [E[3 * j + i] for i in range(3) for j in range(3)]
        return self.mm(F, Et)

    def perp(self, i):
        """A unit vector perpendicular to axis i, chosen on its constant direction."""
        k = self.axes_f[i]
        alt = [1.0, 0.0, 0.0] if abs(k[0]) < 0.9 else [0.0, 1.0, 0.0]
        v = [k[1] * alt[2] - k[2] * alt[1], k[2] * alt[0] - k[0] * alt[2], k[0] * alt[1] - k[1] * alt[0]]
        n = math.sqrt(sum(x * x for x in v))
        return [self.c(x / n) for x in v]

    # -- the solve ---------------------------------------------------------------------------
    def candidates(self, G, q7):
        """The 8 branches (elbow root, z sign, shoulder pair) for one q7 -> list of 7-vectors."""
        k = self.axes
        GB = self.m4(G, self.screw(6, -q7))
        R6 = [GB[4 * i + j] for i in range(3) for j in range(3)]
        W = [w + GB[4 * i + 3] for i, w in enumerate(self.mv(R6, self.wrist))]
        a6 = self.mv(R6, k[5])
        w = [W[i] - self.shoulder[i] for i in range(3)]
        wn = self.norm(w)
        w_hat = [x / wn.maximum(_TINY) for x in w]
        p1 = [self.wrist[i] - self.points[3][i] for i in range(3)]
        p2 = [self.shoulder[i] - self.points[3][i] for i in range(3)]
        out = [None] * 8
        for i4, q4 in enumerate(self.sp3(p1, p2, k[3], wn)):
            R4q = self.rot(k[3], q4)
            v3 = [x + self.points[3][i] - self.shoulder[i] for i, x in enumerate(self.mv(R4q, p1))]
            m3 = self.mv(R4q, k[4])
            v3_hat = [x / self.norm(v3).maximum(_TINY) for x in v3]
            c1, c2, gg = self.dot(m3, v3_hat), self.cos_alpha, self.dot(w_hat, a6)
            det = (1.0 - gg * gg).maximum(_TINY)
            z_par = [((c1 - gg * c2) / det) * w_hat[i] + ((c2 - gg * c1) / det) * a6[i] for i in range(3)]
            perp = self.cross(w_hat, a6)
            perp_hat = [x / self.norm(perp).maximum(_TINY) for x in perp]
            c_mag = (1.0 - self.dot(z_par, z_par)).maximum(0.0).sqrt()
            for isign, sgn in enumerate((1.0, -1.0)):
                z = [z_par[i] + sgn * c_mag * perp_hat[i] for i in range(3)]
                R3 = self.rot_two(v3_hat, m3, w_hat, z)
                t1, t2 = self.sp2(self.mv(R3, k[2]), k[2], k[0], k[1])
                R4m = self.mm(R3, self.rot(k[3], q4))
                Q = self.mtm(R4m, R6)
                q5 = self.sp1(k[5], self.mv(Q, k[5]), k[4])
                Pw = self.mm(self.rot(k[4], -q5), Q)
                uw = self.perp(5)
                q6 = self.sp1(uw, self.mv(Pw, uw), k[5])
                for ipair in range(2):
                    q1, q2 = -t1[ipair], t2[ipair]
                    Pfull = self.mm(self.mm(self.rot(k[1], -q2), self.rot(k[0], -q1)), R3)
                    u = self.perp(2)
                    q3 = self.sp1(u, self.mv(Pfull, u), k[2])
                    out[i4 | (isign << 1) | (ipair << 2)] = [q1, q2, q3, q4, q5, q6, q7]
        return out

    def chain(self, q):
        """Cumulative screw products T_0 = I, T_d = T_{d-1} exp([S_d] q_d), rows 0-2 of each."""
        z, o = self.c(0.0), self.c(1.0)
        T = [o, z, z, z, z, o, z, z, z, z, o, z, z, z, z, o]
        out = [T[:12]]
        for i in range(7):
            T = self.m4(T, self.screw(i, q[i]))
            out.append(T[:12])
        return out

    def pose_error(self, q, target):
        T = self.m4(self.chain(q)[-1] + [self.c(0.0)] * 3 + [self.c(1.0)], self.m_home)
        dp = sum(((T[4 * i + 3] - target[4 * i + 3]) * (T[4 * i + 3] - target[4 * i + 3]) for i in range(1, 3)),
                 (T[3] - target[3]) * (T[3] - target[3]))
        fro = self.c(0.0)
        for i in range(3):
            for j in range(3):
                re = sum((T[4 * i + kk] * target[4 * j + kk] for kk in range(1, 3)), T[4 * i] * target[4 * j])
                re = re - 1.0 if i == j else re
                fro = fro + re * re
        return dp.sqrt() + (fro * 0.5).sqrt()


@algorithm(inputs=(("target", lambda o: 16), ("q7", lambda o: 1)), outputs=("q", "err"))
def analytic_ik_candidates(robot, target, q7, geom=()):
    """GRiM ``analytic_ik``'s eight branches for one redundancy value ``q7``: ``q`` (8 x 7,
    branch ``elbow | sign << 1 | pair << 2``) and each one's pose error (position norm plus the
    Frobenius rotation error / sqrt 2), against ``target`` (4 x 4 row-major). ``geom`` is the
    95-scalar ArmGeom, folded in. Non-finite candidates and limits are the caller's to reject."""
    arm = _Arm(robot.t, geom)
    G = arm.m4(list(target), arm.m_home_inv)
    qs = arm.candidates(G, q7[0])
    return [x for q in qs for x in q], [arm.pose_error(q, target) for q in qs]


@algorithm(inputs=(("q", lambda o: 7),), outputs=("chain",))
def analytic_ik_chain(robot, q, geom=()):
    """The cumulative screw products (8 x rows 0-2 of a 4 x 4, depth 0 = identity) that place
    ``analytic_ik``'s home-frame collision spheres at ``q``."""
    arm = _Arm(robot.t, geom)
    return [x for T in arm.chain(q) for x in T]
