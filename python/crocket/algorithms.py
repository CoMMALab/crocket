"""The algorithms, ported from RBDReference (which stays the oracle), written once over the
traced spatial types. Joint loops unroll at trace time; mimic joints feed their mimicked column
scaled by their multiplier, as in RBDReference.

Outputs: poses are ``[t (3), R column-major (9)]`` per frame; the Jacobian is row-major
``6 x n_q``, rows ``(vx, vy, vz, wx, wy, wz)``, LOCAL_WORLD_ALIGNED at the frame origin;
matrices are ``n_q x n_q`` column-major.
"""

from __future__ import annotations

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
