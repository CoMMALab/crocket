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


@algorithm(inputs=("q",), outputs=("pose", "jacobian"))
def frame_jacobian(robot, q, frame=None):
    X0 = _poses(robot, q)
    Xe = _frame(robot, X0, frame)
    p = Xe.translation
    f = robot.frame(frame)
    zero = Scalar.const(robot.t, 0.0)
    J = [[zero] * robot.n_q for _ in range(6)]
    for j in (robot.ancestors(f.parent) if f.parent >= 0 else []):
        col = (X0[j].inv() @ robot.S(j)).at(p)  # world-aligned twist at the frame origin
        for row, k in enumerate((3, 4, 5, 0, 1, 2)):
            J[row][robot.col(j)] = J[row][robot.col(j)] + col[k] * robot.scale(j)
    return _pose_out(Xe), J


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
