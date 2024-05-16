import os
import copy

import numpy as np
import scipy.stats
from scipy.integrate import solve_bvp

import matplotlib.pyplot as plt

phi_list = np.empty(0)
fk_list = np.empty(0)
h_list = np.empty(0)
c_list = np.empty(0)
f_traj = np.empty(0)
lam_list = np.empty(0)
dfkdxdt_list = np.empty(0)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


def pdf(x, weights, dists, dx, dy):
    result = np.zeros(shape=x.shape[0])

    for w, dist in zip(weights, dists):
        result += w * dist.pdf(x)

    result /= np.sum(result * dx * dy)
    return result


def dyn_random(_, weights, dists):
    ind_dist = np.random.choice(a=np.arange(len(weights)), size=1, p=weights)[0]
    # print(ind_dist)
    dist = dists[ind_dist]
    return dist.rvs()


def xdot(_, ut):
    return ut


def next_state(xt, ut, dt):
    k1 = dt * xdot(xt, ut)
    k2 = dt * xdot(xt + k1 / 2.0, ut)
    k3 = dt * xdot(xt + k2 / 2.0, ut)
    k4 = dt * xdot(xt + k3, ut)

    xt_new = xt + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return xt_new


# def barr(x_traj):
#     x_traj_barr = np.zeros_like(x_traj)

#     for xt in x_traj:
#         x1, x2 = xt
#         x1_barr = min(0, x-0)**2+max()


def main_p1(
    dt=0.1,
    T=10,
    # init_u_traj=np.zeros(shape=(63, 2)),
    # x0=np.array([0.0, 0.0, np.pi / 2.0]),
    x0=np.array([0.3, 0.3]),
    # Q_x=np.diag([0.01, 0.01, 2.0]),
    q=0.01,
    R_u=np.diag([0.0, 0.0]),
    # P1=np.diag([20.0, 20.0, 5.0]),
    P1=np.diag([2.0, 2.0]),
    # Q_z=np.diag([5.0, 5.0, 1.0]),
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
    tsteps = tlist.shape[0]

    weights = np.array([w1, w2, w3])
    means = np.array([mu1, mu2, mu3])
    covs = np.array([cov1, cov2, cov3])

    N = N_grid + 1

    dx = dy = 1.0 / N_grid

    dists = []
    for mu, cov in zip(means, covs):
        dist = scipy.stats.multivariate_normal(mean=mu, cov=cov)
        dists.append(dist)

    kdim_x = np.arange(K_per_dim)
    kdim_y = np.arange(K_per_dim)

    ks_x, ks_y = np.meshgrid(kdim_x, kdim_y)
    ks_xy = np.array([ks_x.ravel(), ks_y.ravel()]).T

    L_list = np.array([1, 1], dtype="float")

    grid_x, grid_y = np.meshgrid(
        np.linspace(0, L_list[0], N),
        np.linspace(0, L_list[1], N),
    )
    grids_xy = np.array([grid_x.ravel(), grid_y.ravel()]).T
    pdf_gt = pdf(
        x=grids_xy,
        weights=weights,
        dists=dists,
        dx=dx,
        dy=dy,
    )

    phi_list = np.zeros(ks_xy.shape[0])
    fk_list = np.zeros(shape=(ks_xy.shape[0], N**2))
    h_list = np.zeros(ks_xy.shape[0])

    for i, k_vec in enumerate(ks_xy):
        fk_vals = np.prod(np.cos(k_vec * np.pi / L_list * grids_xy), axis=1)
        hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
        fk_vals /= hk
        phi_k = np.sum(pdf_gt * fk_vals * dx * dy)

        h_list[i] = hk
        fk_list[i, :] = fk_vals
        phi_list[i] = phi_k

    c_list = np.zeros(ks_xy.shape[0])
    f_traj = np.zeros((ks_xy.shape[0], tsteps))
    lam_list = np.power(1.0 + np.linalg.norm(ks_xy, axis=1), -3.0 / 2.0)
    dfkdxdt_list = np.zeros((ks_xy.shape[0], 2))

    def dyn(xt, ut):
        # xdot = np.zeros(3)  # replace this
        # theta = xt[2]
        # u1 = ut[0]
        # u2 = ut[1]
        # x1dot = np.cos(theta) * u1
        # x2dot = np.sin(theta) * u1
        # x3dot = u2

        # xdot = np.array([x1dot, x2dot, x3dot])
        xdot = copy.deepcopy(ut)
        return xdot

    def get_A(t, xt, ut):
        # theta = xt[2]
        # u1 = ut[0]
        A_mat = np.zeros((2, 2))  # replace this
        # A_mat[0, 2] = -np.sin(theta) * u1
        # A_mat[1, 2] = np.cos(theta) * u1
        return A_mat

    def get_B(t, xt, ut):
        # theta = xt[2]
        # B_mat = np.zeros((2, 2))  # replace this
        B_mat = np.eye(2)
        # B_mat[0, 0] = np.cos(theta)
        # B_mat[1, 0] = np.sin(theta)
        # B_mat[2, 1] = 1
        return B_mat

    def get_xd(t):
        xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])
        return xd

    def step(xt, ut):
        # xt_new = xt + dt * dyn(xt, ut)  # recommended: replace it with RK4 integration
        k1 = dt * dyn(xt, ut)
        k2 = dt * dyn(xt + k1 / 2.0, ut)
        k3 = dt * dyn(xt + k2 / 2.0, ut)
        k4 = dt * dyn(xt + k3, ut)

        xt_new = xt + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return xt_new

    def traj_sim(x0, ulist):
        # tsteps = ulist.shape[0]
        x_traj = np.zeros((tsteps + 1, x0.shape[0]))
        x_traj[0, :] = x0
        xt = copy.deepcopy(x0)
        for t in range(tsteps):
            xt_new = step(xt, ulist[t])
            x_traj[t + 1] = copy.deepcopy(xt_new)
            xt = copy.deepcopy(xt_new)
        return x_traj

    # def loss(t, xt, ut):
    #     # xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])  # desired system state at time t
    #     xd = get_xd(t)

    #     x_diff = xt - xd
    #     # x_loss = 0.0  # replace this
    #     x_loss = x_diff.T @ Q_x @ x_diff
    #     # u_loss = 0.0  # replace this
    #     u_loss = ut.T @ R_u @ ut

    #     return x_loss + u_loss

    def dldx(t, xt, ut):
        # xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])
        # xd = get_xd(t)
        # qlist = np.diag(Q_x)
        t_ind = np.where(tlist == t)[0][0]
        dvec = np.zeros(2)  # replace this
        for lam_k, ck, phi_k, hk, k_vec, fk_traj, dfkdxdt in zip(
            lam_list, c_list, phi_list, h_list, ks_xy, f_traj, dfkdxdt_list
        ):
            # ax_val
            # k1, k2 = k_vec
            # d_fk = np.zeros(2)
            # for xt in x_traj:
            fk_t = fk_traj[t_ind]
            # dfkdxdt = dfkdxdt_list
            # print(dfkdxdt)
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

            # dvec += lam_k / (T**2) * (dfkdx * ck * T - 2 * T * dfkdx * phi_k)
            # dvec += lam_k / (T**2) * (fk_t * dfkdxdt)
            dvec += q * lam_k * (2 * (ck - phi_k) * 1 / T * dfkdx)

        # dx = xt - xd
        # dvec = 2 * qlist * dx
        return dvec

    def dldu(t, xt, ut):
        dvec = np.zeros(2)  # replace this
        rlist = np.diag(R_u)
        dvec = rlist * ut
        return dvec

    def func_J(x_traj, u_traj):
        J_val = 0

        # print(c_list)
        J_val += q * np.sum(lam_list * np.square(c_list - phi_list))
        # print(u_traj.T @ R_u @ u_traj)
        for ut in u_traj:
            # print(ut)
            J_val += ut.T @ R_u @ ut * dt
        # for xt, ut, t in zip(x_traj[:, :], u_traj, tlist):
        #     l_val = loss(t, xt, ut)
        #     J_val += l_val

        # xd_T = get_xd(tlist[-1])
        # xT = x_traj[-1]
        # dx_T = xT - xd_T

        # J_val += dx_T.T @ P1 @ dx_T

        return J_val

    def ilqr_iter(x0, u_traj):
        """
        :param x0: initial state of the system
        :param u_traj: current estimation of the optimal control trajectory
        :return: the descent direction for the control
        """
        # forward simulate the state trajectory
        x_traj = traj_sim(x0, u_traj)

        c_list = np.zeros(ks_xy.shape[0])
        f_traj = np.zeros([ks_xy.shape[0], x_traj.shape[0]])
        for i, (k_vec, hk) in enumerate(zip(ks_xy, h_list)):
            fk_vals = np.prod(np.cos(k_vec * np.pi / L_list * x_traj), axis=1)
            dfkdxdt = np.zeros(2)

            hk = h_list[i]
            fk_vals /= hk

            ck = np.sum(fk_vals) * dt / T

            for xt in x_traj:
                dfkdx = (
                    -1
                    / hk
                    * k_vec
                    * np.sin(k_vec * np.pi * xt)
                    * np.cos(k_vec[::-1] * np.pi * xt[::-1])
                )
                dfkdxdt += dfkdx * dt

            # print(fk_vals.shape)
            # fk_vals_gt = fk_list[i, :]

            dfkdxdt_list[i, :] = copy.deepcopy(dfkdxdt)
            f_traj[i, :] = fk_vals
            c_list[i] = ck

        # print("in loop: ", c_list)
        # print(dfkdxdt_list)

        # compute other variables needed for specifying the dynamics of z(t) and p(t)
        A_list = np.zeros((tsteps, 2, 2))
        B_list = np.zeros((tsteps, 2, 2))
        a_list = np.zeros((tsteps, 2))
        b_list = np.zeros((tsteps, 2))
        for t_idx, t in np.ndenumerate(tlist):
            # t = t_idx * dt
            A_list[t_idx] = get_A(t, x_traj[t_idx], u_traj[t_idx])
            B_list[t_idx] = get_B(t, x_traj[t_idx], u_traj[t_idx])
            a_list[t_idx] = dldx(t, x_traj[t_idx], u_traj[t_idx])
            b_list[t_idx] = dldu(t, x_traj[t_idx], u_traj[t_idx])

        xd_T = get_xd(tlist[-1])  # desired terminal state
        xT = x_traj[-1, :]
        xT_2 = x_traj[int(x_traj.shape[0] / 2), :]

        plist = np.diag(P1)
        # p1 = np.zeros(3)  # replace it to be the terminal condition p(T)

        p1 = 2 * plist * (xT - mu2) * (xT_2 - mu3)
        # p1 = plist * xT
        p1 = np.zeros(2)

        def zp_dyn(t, zp):
            zt = zp[:2]
            pt = zp[2:]

            t_idx = int(t / dt)
            At = A_list[t_idx]
            Bt = B_list[t_idx]
            at = a_list[t_idx]
            bt = b_list[t_idx]

            # M_11 = np.zeros((3,3))  # replace this
            M_11 = At
            M_12 = np.zeros((2, 2))  # replace this
            M_21 = np.zeros((2, 2))  # replace this
            # M_22 = np.zeros((3,3))  # replace this
            M_22 = -At.T
            dyn_mat = np.block([[M_11, M_12], [M_21, M_22]])

            # m_1 = np.zeros(3)  # replace this
            m_1 = -Bt @ np.linalg.inv(R_v.T) @ (pt.T @ Bt + bt.T)
            # m_2 = np.zeros(3)  # replace this
            m_2 = -at - zt @ Q_z
            dyn_vec = np.hstack([m_1, m_2])

            return dyn_mat @ zp + dyn_vec

        # this will be the actual dynamics function you provide to solve_bvp,
        # it takes in a list of time steps and corresponding [z(t), p(t)]
        # and returns a list of [zdot(t), pdot(t)]
        def zp_dyn_list(t_list, zp_list):
            list_len = len(t_list)
            zp_dot_list = np.zeros((4, list_len))
            for _i in range(list_len):
                zp_dot_list[:, _i] = zp_dyn(t_list[_i], zp_list[:, _i])
            return zp_dot_list

        # boundary condition (inputs are [z(0),p(0)] and [z(T),p(T)])
        def zp_bc(zp_0, zp_T):
            # return np.zeros(6)  # replace this
            z0 = zp_0[:2]
            p0 = zp_0[2:]

            zT = zp_T[:2]
            pT = zp_T[2:]

            bc = np.zeros(4)
            bc[:2] = z0
            bc[2:] = np.abs(pT - p1)
            # print(bc)

            return bc

        ### The solver will say it does not converge, but the returned result
        ### is numerically accurate enough for our use
        # zp_traj = np.zeros((tsteps,6))  # replace this by using solve_bvp
        res = solve_bvp(
            zp_dyn_list,
            zp_bc,
            tlist,
            np.zeros(shape=(4, tsteps)),
            verbose=1,
            max_nodes=100,
        )
        # zp_traj = np.zeros(shape=(tsteps, 6))
        # # print(res.x.shape)

        # for i in range(6):
        #     f = sitr.interp1d(res.x, res.y.T[:, i])
        #     zp_traj[:, i] = f(tlist)
        zp_traj = res.sol(tlist).T

        # print(zp_traj)

        z_traj = zp_traj[:, :2]
        p_traj = zp_traj[:, 2:]

        v_traj = np.zeros((tsteps, 2))
        for _i in range(tsteps):
            At = A_list[_i]
            Bt = B_list[_i]
            at = a_list[_i]
            bt = b_list[_i]

            zt = z_traj[_i]
            pt = p_traj[_i]

            # vt = np.zeros(2)  # replace this
            vt = -np.linalg.inv(R_v.T) @ (pt.T @ Bt + bt.T)
            v_traj[_i, :] = vt

        return v_traj, c_list

    # Start iLQR iterations here

    u_traj = init_u_traj.copy()
    Jlist = np.array([func_J(traj_sim(x0, u_traj), u_traj)])

    for iter in range(max_iter):
        print(f"Iteration {iter} ...")
        # forward simulate the current trajectory
        x_traj = traj_sim(x0, u_traj)

        # visualize the current trajectory
        # fig, axs = plt.subplots(1, 2)
        # axs[0].set_title(f"Iter: {iter}")
        # axs[0].set_aspect("equal")
        # axs[0].set_xlim(-0.2, 4.2)
        # axs[0].set_ylim(-0.2, 2.2)
        # axs[0].plot(x_traj[:, 0], x_traj[:, 1], linestyle="-", color="C0")

        # axs[1].plot(tlist, u_traj[:, 0], linestyle="-", label="u_1")
        # axs[1].plot(tlist, u_traj[:, 1], linestyle="-", label="u_2")
        # axs[1].legend()
        # plt.show()
        # plt.close()

        # get descent direction

        # print("before: ", c_list)
        v_traj, c_list = ilqr_iter(x0, u_traj)
        # print("after: ", c_list)
        # print(v_traj)

        # Armijo line search parameters
        gamma = copy.deepcopy(gamma_0)  # initial step size
        alpha = 1e-4
        beta = 0.5

        # print(-v_traj.T @ v_traj)

        ### Implement Armijo line search here to update step size gamma
        while func_J(x_traj, u_traj + gamma * v_traj) > func_J(
            x_traj, u_traj
        ) + alpha * gamma * np.abs(np.trace(-v_traj.T @ v_traj)):
            gamma *= beta

        # gamma = 0

        # update control for the next iteration
        u_traj += gamma * v_traj
        Jlist = np.hstack([Jlist, func_J(x_traj, u_traj)])
        print(f"Gamma = {gamma}, J = {Jlist[-1]}")
        if gamma < 1e-5:
            break

    # print(Jlist)
    init_x_traj = traj_sim(x0, init_u_traj)
    # xd_traj = np.zeros(shape=(tsteps, x0.shape[0]))

    # for i in range(tsteps):
    #     xd_traj[i, :] = get_xd(tlist[i])

    # print(x_traj.shape, u_traj.shape)
    # print(xd_traj)

    _, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].set_title("State Trajectory")
    axs[0].set_xlabel(r"$x(t)$")
    axs[0].set_ylabel(r"$y(t)$")
    axs[0].imshow(
        pdf_gt.reshape(N, N),
        origin="lower",
        cmap="Reds",
        extent=[0, 1, 0, 1],
    )
    axs[0].plot(
        init_x_traj[:, 0],
        init_x_traj[:, 1],
        linestyle="-.",
        label="Initial Trajectory",
    )
    axs[0].plot(
        x_traj[:, 0],
        x_traj[:, 1],
        linestyle="-",
        color="k",
        label="Converged Trajectory",
    )
    # axs[0].plot(
    #     xd_traj[:, 0],
    #     xd_traj[:, 1],
    #     linestyle="--",
    #     label="Desired Trajectory",
    # )
    # axs[0].set_aspect("equal")
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
    # plt.show()
    plt.savefig(fname=os.path.join(RESULTS_DIR, fig_filename), format="png", dpi=300)
    # plt.close()


if __name__ == "__main__":
    u_traj = np.tile(np.array([-10, 10]), reps=100)

    # p1_main_old(
    #     w1=0.5,
    #     mu1=np.array([0.35, 0.38]),
    #     cov1=np.array([[0.01, 0.004], [0.004, 0.01]]),
    #     w2=0.2,
    #     mu2=np.array([0.68, 0.25]),
    #     cov2=np.array([[0.005, -0.003], [-0.003, 0.005]]),
    #     w3=0.3,
    #     mu3=np.array([0.56, 0.64]),
    #     cov3=np.array([[0.008, 0.0], [0.0, 0.004]]),
    #     u_traj=u_traj,
    #     N_grid=100,
    #     K_per_dim=10,
    #     dt=0.1,
    #     T=10,
    # )

    #     main_p3(
    #     w1=w1,
    #     w2=w2,
    #     w3=w3,
    #     mu1=mu1,
    #     mu2=mu2,
    #     mu3=mu3,
    #     cov1=cov1,
    #     cov2=cov2,
    #     cov3=cov3,
    #     init_u_traj=u_traj,
    #     # u_traj=u_traj,
    #     N_grid=100,
    #     K_per_dim=10,
    #     dt=0.1,
    #     T=10,
    # )
