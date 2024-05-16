import os
import numpy as np
import scipy.stats
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


def pdf(x, weights, dists, dx, dy):
    result = sum(w * dist.pdf(x) for w, dist in zip(weights, dists))
    result /= np.sum(result * dx * dy)
    return result


def dyn_random(_, weights, dists):
    ind_dist = np.random.choice(len(weights), p=weights)
    dist = dists[ind_dist]
    return dist.rvs()


def xdot(_, ut):
    return ut


def next_state(xt, ut, dt):
    k1 = dt * xdot(xt, ut)
    k2 = dt * xdot(xt + k1 / 2.0, ut)
    k3 = dt * xdot(xt + k2 / 2.0, ut)
    k4 = dt * xdot(xt + k3, ut)
    return xt + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def main_p1(
    dt=0.1,
    T=10,
    x0=np.array([0.3, 0.3]),
    q=0.01,
    R_u=np.diag([0.0, 0.0]),
    P1=np.diag([2.0, 2.0]),
    Q_z=np.diag([0.001, 0.001]),
    R_v=np.diag([0.01, 0.01]),
    w1=0.5,
    mu1=np.array([0.35, 0.38]),
    cov1=np.array([[0.01, 0.004], [0.004, 0.01]]),
    w2=0.2,
    mu2=np.array([0.68, 0.25]),
    cov2=np.array([[0.005, -0.003], [-0.003, 0.005]]),
    w3=0.3,
    mu3=np.array([0.56, 0.64]),
    cov3=np.array([[0.008, 0.0], [0.0, 0.004]]),
    init_u_traj=np.tile(np.array([-0.1, 0.1]), reps=100),
    N_grid=1000,
    K_per_dim=10,
    gamma_0=0.001,
    max_iter=100,
    fig_filename="example.png",
):
    tlist = np.arange(0, T, dt)
    tsteps = len(tlist)

    weights = np.array([w1, w2, w3])
    means = np.array([mu1, mu2, mu3])
    covs = np.array([cov1, cov2, cov3])

    N = N_grid + 1
    dx = dy = 1.0 / N_grid

    dists = [
        scipy.stats.multivariate_normal(mean=mu, cov=cov)
        for mu, cov in zip(means, covs)
    ]

    kdim_x, kdim_y = np.arange(K_per_dim), np.arange(K_per_dim)
    ks_x, ks_y = np.meshgrid(kdim_x, kdim_y)
    ks_xy = np.column_stack([ks_x.ravel(), ks_y.ravel()])

    L_list = np.array([1, 1], dtype=float)
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, L_list[0], N), np.linspace(0, L_list[1], N)
    )
    grids_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    pdf_gt = pdf(grids_xy, weights, dists, dx, dy)

    fk_list = np.zeros((ks_xy.shape[0], N**2))
    h_list = np.zeros(ks_xy.shape[0])
    phi_list = np.zeros(ks_xy.shape[0])

    for i, k_vec in enumerate(ks_xy):
        fk_vals = np.prod(np.cos(k_vec * np.pi / L_list * grids_xy), axis=1)
        hk = np.sqrt(np.sum(fk_vals**2) * dx * dy)
        fk_vals /= hk
        fk_list[i, :] = fk_vals
        h_list[i] = hk
        phi_list[i] = np.sum(pdf_gt * fk_vals * dx * dy)

    lam_list = (1.0 + np.linalg.norm(ks_xy, axis=1)) ** (-3.0 / 2.0)
    c_list = np.zeros(ks_xy.shape[0])
    f_traj = np.zeros((ks_xy.shape[0], tsteps))
    lam_list = np.power(1.0 + np.linalg.norm(ks_xy, axis=1), -3.0 / 2.0)
    dfkdxdt_list = np.zeros((ks_xy.shape[0], 2))

    def dyn(xt, ut):
        return ut

    def get_A(_, __, ___):
        return np.zeros((2, 2))

    def get_B(_, __, ___):
        return np.eye(2)

    def get_xd(t):
        return np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])

    def step(xt, ut):
        k1 = dt * dyn(xt, ut)
        k2 = dt * dyn(xt + k1 / 2.0, ut)
        k3 = dt * dyn(xt + k2 / 2.0, ut)
        k4 = dt * dyn(xt + k3, ut)
        return xt + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def traj_sim(x0, ulist):
        x_traj = np.zeros((tsteps + 1, x0.shape[0]))
        x_traj[0, :] = x0
        xt = x0.copy()
        for t in range(tsteps):
            xt = step(xt, ulist[t])
            x_traj[t + 1] = xt
        return x_traj

    def dldx(t, xt, _):
        t_ind = np.where(tlist == t)[0][0]
        dvec = np.zeros(2)
        for lam_k, ck, phi_k, hk, k_vec in zip(
            lam_list, c_list, phi_list, h_list, ks_xy
        ):
            fk_t = f_traj[:, t_ind]
            dfkdx = (
                -1.0
                / hk
                * (
                    k_vec
                    * np.pi
                    * np.sin(k_vec * np.pi * xt)
                    * np.cos(k_vec[::-1] * np.pi * xt[::-1])
                )
            )
            dvec += q * lam_k * (2 * (ck - phi_k) / T * dfkdx)
        return dvec

    def dldu(_, __, ut):
        return np.diag(R_u) * ut

    def func_J(x_traj, u_traj):
        J_val = q * np.sum(lam_list * (c_list - phi_list) ** 2)
        for ut in u_traj:
            J_val += ut @ R_u @ ut * dt
        return J_val

    def ilqr_iter(x0, u_traj):
        x_traj = traj_sim(x0, u_traj)
        global c_list, f_traj
        c_list = np.zeros(ks_xy.shape[0])
        f_traj = np.zeros((ks_xy.shape[0], tsteps + 1))
        for i, k_vec in enumerate(ks_xy):
            fk_vals = np.prod(np.cos(k_vec * np.pi / L_list * x_traj), axis=1)
            fk_vals /= h_list[i]
            ck = np.sum(fk_vals) * dt / T
            dfkdxdt = np.sum(
                [
                    -1
                    / h_list[i]
                    * k_vec
                    * np.sin(k_vec * np.pi * xt)
                    * np.cos(k_vec[::-1] * np.pi * xt[::-1])
                    * dt
                    for xt in x_traj
                ],
                axis=0,
            )
            f_traj[i, :] = fk_vals
            c_list[i] = ck
            # dfkdxdt_list[i, :] = dfkdxdt

        A_list = np.array([get_A(t, x_traj[i], u_traj[i]) for i, t in enumerate(tlist)])
        B_list = np.array([get_B(t, x_traj[i], u_traj[i]) for i, t in enumerate(tlist)])
        a_list = np.array([dldx(t, x_traj[i], u_traj[i]) for i, t in enumerate(tlist)])
        b_list = np.array([dldu(t, x_traj[i], u_traj[i]) for i, t in enumerate(tlist)])

        p1 = np.zeros(2)

        def zp_dyn(_, zp):
            zt, pt = zp[:2], zp[2:]
            t_idx = int(_ / dt)
            At, Bt, at, bt = A_list[t_idx], B_list[t_idx], a_list[t_idx], b_list[t_idx]
            M_11, M_22 = At, -At.T
            dyn_mat = np.block([[M_11, np.zeros((2, 2))], [np.zeros((2, 2)), M_22]])
            m_1 = -Bt @ np.linalg.inv(R_v.T) @ (pt.T @ Bt + bt.T)
            m_2 = -at - zt @ Q_z
            return dyn_mat @ zp + np.hstack([m_1, m_2])

        def zp_dyn_list(t_list, zp_list):
            return np.array([zp_dyn(t, zp) for t, zp in zip(t_list, zp_list.T)]).T

        def zp_bc(zp_0, zp_T):
            z0, p0 = zp_0[:2], zp_0[2:]
            zT, pT = zp_T[:2], zp_T[2:]
            return np.hstack([z0, np.abs(pT - p1)])

        res = solve_bvp(
            zp_dyn_list, zp_bc, tlist, np.zeros((4, tsteps)), verbose=1, max_nodes=100
        )
        zp_traj = res.sol(tlist).T

        z_traj, p_traj = zp_traj[:, :2], zp_traj[:, 2:]
        v_traj = np.array(
            [
                -np.linalg.inv(R_v.T) @ (pt.T @ Bt + bt.T)
                for pt, Bt, bt in zip(p_traj, B_list, b_list)
            ]
        )
        return v_traj

    u_traj = init_u_traj.copy()
    Jlist = np.array([func_J(traj_sim(x0, u_traj), u_traj)])

    for iter in range(max_iter):
        print(f"Iteration {iter} ...")
        x_traj = traj_sim(x0, u_traj)
        v_traj = ilqr_iter(x0, u_traj)

        gamma = gamma_0
        alpha = 1e-4
        beta = 0.5

        while func_J(x_traj, u_traj + gamma * v_traj) > func_J(
            x_traj, u_traj
        ) + alpha * gamma * np.abs(np.trace(-v_traj.T @ v_traj)):
            gamma *= beta

        u_traj += gamma * v_traj
        Jlist = np.hstack([Jlist, func_J(x_traj, u_traj)])
        print(gamma)
        if gamma < 1e-5:
            break

    init_x_traj = traj_sim(x0, init_u_traj)

    _, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].set_title("State Trajectory")
    axs[0].set_xlabel(r"$x(t)$")
    axs[0].set_ylabel(r"$y(t)$")
    axs[0].imshow(
        pdf_gt.reshape(N, N), origin="lower", cmap="Reds", extent=[0, 1, 0, 1]
    )
    axs[0].plot(
        init_x_traj[:, 0], init_x_traj[:, 1], linestyle="-.", label="Initial Trajectory"
    )
    axs[0].plot(
        x_traj[:, 0],
        x_traj[:, 1],
        linestyle="-",
        color="k",
        label="Converged Trajectory",
    )
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend()

    axs[1].set_title("Optimal Control")
    axs[1].set_xlabel(r"Time $[t]$")
    axs[1].set_ylabel(r"$\vec{u}(t)$")
    axs[1].plot(tlist, u_traj[:, 0], label=r"$u_1$")
    axs[1].plot(tlist, u_traj[:, 1], label=r"$u_2$")
    axs[1].legend()

    axs[2].set_title("Objective Value")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Objective")
    axs[2].plot(Jlist)

    plt.savefig(fname=os.path.join(RESULTS_DIR, fig_filename), format="png", dpi=300)


if __name__ == "__main__":
    u_traj = np.tile(np.array([-10, 10]), reps=100)
    main_p1(
        w1=0.5,
        mu1=np.array([0.35, 0.38]),
        cov1=np.array([[0.01, 0.004], [0.004, 0.01]]),
        w2=0.2,
        mu2=np.array([0.68, 0.25]),
        cov2=np.array([[0.005, -0.003], [-0.003, 0.005]]),
        w3=0.3,
        mu3=np.array([0.56, 0.64]),
        cov3=np.array([[0.008, 0.0], [0.0, 0.004]]),
        init_u_traj=u_traj,
        N_grid=100,
        K_per_dim=10,
        dt=0.1,
        T=10,
    )
