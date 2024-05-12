import copy
import os
import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import sympy as sp
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

A = sp.symbols("A")
display(A)

############### Problem 2 ###############


def opt_func(x1, x2):
    f = 0.26 * (x1**2 + x2**2) - 0.46 * x1 * x2
    return f


def gradient(x1, x2):
    dJdx1 = 0.52 * x1 - 0.46 * x2
    dJdx2 = 0.52 * x2 - 0.46 * x1

    return np.array([dJdx1, dJdx2])


def update_plot(frame, line: Line2D, traj):
    line.set_xdata(traj[:frame, 0])
    line.set_ydata(traj[:frame, 1])

    return (line,)


def main_p2(
    n_iter: int = 100,
    start: tuple[float, float] = (-4, -2),
    alpha: float = 1e-4,
    beta: float = 0.5,
):
    x_range = np.linspace(-5, 5, 1000)
    y_range = np.linspace(-5, 5, 1000)

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = opt_func(x1=x_grid, x2=y_grid)

    gamma_0 = 1
    # alpa = 1e-4
    # beta = 0.5
    # x0 = np.array([-4, -2], dtype="float")
    x0 = np.asarray(start, dtype="float")
    x = copy.deepcopy(x0)

    traj = np.zeros(shape=(n_iter + 1, 2))
    traj[0, :] = x

    print("Simulating ...")

    for i in range(n_iter):
        gamma = gamma_0
        z = -gradient(*x)

        while (opt_func(*(x + gamma * z))) > opt_func(*x) + alpha * gamma * np.dot(
            gradient(*x), z
        ):
            gamma *= beta

        # print(gamma)

        x += gamma * z
        traj[i + 1, :] = x

    print("Plotting results ...")

    plt.imshow(
        z_grid,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="Blues_r",
        norm=mcolors.LogNorm(),
    )
    plt.colorbar()
    # plt.contourf(x_grid, y_grid, z_grid, cmap="Blues_r", norm=mcolors.LogNorm())
    plt.plot(traj[:, 0], traj[:, 1], color="m", linewidth=1, label="Iterations")
    plt.scatter(*traj[-1, :], color="m", label="Converged estimation")
    plt.scatter(*x0, color="k", label="Initial guess")
    plt.legend()

    plt.savefig(fname=os.path.join(RESULTS_DIR, "part2.png"), format="png", dpi=300)

    fig, ax = plt.subplots()
    (line,) = plt.plot(*x0, color="m", linewidth=1)
    plt.imshow(
        z_grid,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="Blues_r",
        norm=mcolors.LogNorm(),
    )
    plt.colorbar()

    animate = FuncAnimation(
        fig=fig,
        func=update_plot,
        frames=np.arange(1, n_iter + 2),
        fargs=(line, traj),
        interval=5.0 / (n_iter + 1) * 1e3,
        blit=True,
    )

    animate.save(filename=os.path.join(RESULTS_DIR, "part2.mp4"), writer="ffmpeg")

    print("Simulation result saved")


############### Problem 3 ###############


def main_p3(
    dt=0.1,
    T=2.0 * np.pi,
    init_u_traj=np.zeros(shape=(63, 2)),
    x0=np.array([0.0, 0.0, np.pi / 2.0]),
    Q_x=np.diag([10.0, 10.0, 2.0]),
    R_u=np.diag([4.0, 2.0]),
    P1=np.diag([20.0, 20.0, 5.0]),
    Q_z=np.diag([5.0, 5.0, 1.0]),
    R_v=np.diag([2.0, 1.0]),
    fig_filename="example.png",
):
    # dt = 0.1
    # T = 2.0 * np.pi
    # x0 = np.array([0.0, 0.0, np.pi / 2.0])
    tlist = np.arange(0, T, dt)
    tsteps = tlist.shape[0]
    # init_u_traj = np.tile(np.array([1.5, -1.0]), reps=(tsteps, 1))

    # Q_x = np.diag([10.0, 10.0, 2.0])
    # R_u = np.diag([4.0, 2.0])
    # P1 = np.diag([20.0, 20.0, 5.0])

    # Q_z = np.diag([5.0, 5.0, 1.0])
    # R_v = np.diag([2.0, 1.0])

    def dyn(xt, ut):
        # xdot = np.zeros(3)  # replace this
        theta = xt[2]
        u1 = ut[0]
        u2 = ut[1]
        x1dot = np.cos(theta) * u1
        x2dot = np.sin(theta) * u1
        x3dot = u2

        xdot = np.array([x1dot, x2dot, x3dot])
        return xdot

    def get_A(t, xt, ut):
        theta = xt[2]
        u1 = ut[0]
        A_mat = np.zeros((3, 3))  # replace this
        A_mat[0, 2] = -np.sin(theta) * u1
        A_mat[1, 2] = np.cos(theta) * u1
        return A_mat

    def get_B(t, xt, ut):
        theta = xt[2]
        B_mat = np.zeros((3, 2))  # replace this
        B_mat[0, 0] = np.cos(theta)
        B_mat[1, 0] = np.sin(theta)
        B_mat[2, 1] = 1
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
        tsteps = ulist.shape[0]
        x_traj = np.zeros((tsteps + 1, 3))
        x_traj[0, :] = x0
        xt = copy.deepcopy(x0)
        for t in range(tsteps):
            xt_new = step(xt, ulist[t])
            x_traj[t + 1] = copy.deepcopy(xt_new)
            xt = copy.deepcopy(xt_new)
        return x_traj

    def loss(t, xt, ut):
        # xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])  # desired system state at time t
        xd = get_xd(t)

        x_diff = xt - xd
        # x_loss = 0.0  # replace this
        x_loss = x_diff.T @ Q_x @ x_diff
        # u_loss = 0.0  # replace this
        u_loss = ut.T @ R_u @ ut

        return x_loss + u_loss

    def dldx(t, xt, ut):
        # xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])
        xd = get_xd(t)
        qlist = np.diag(Q_x)

        # dvec = np.zeros(3)  # replace this
        dx = xt - xd
        dvec = 2 * qlist * dx
        return dvec

    def dldu(t, xt, ut):
        # dvec = np.zeros(2)  # replace this
        rlist = np.diag(R_u)
        dvec = 2 * rlist * ut
        return dvec

    def func_J(x_traj, u_traj):
        J_val = 0

        for xt, ut, t in zip(x_traj[:, :], u_traj, tlist):
            l_val = loss(t, xt, ut)
            J_val += l_val

        xd_T = get_xd(tlist[-1])
        xT = x_traj[-1]
        dx_T = xT - xd_T

        J_val += dx_T.T @ P1 @ dx_T

        return J_val

    def ilqr_iter(x0, u_traj):
        """
        :param x0: initial state of the system
        :param u_traj: current estimation of the optimal control trajectory
        :return: the descent direction for the control
        """
        # forward simulate the state trajectory
        x_traj = traj_sim(x0, u_traj)

        # compute other variables needed for specifying the dynamics of z(t) and p(t)
        A_list = np.zeros((tsteps, 3, 3))
        B_list = np.zeros((tsteps, 3, 2))
        a_list = np.zeros((tsteps, 3))
        b_list = np.zeros((tsteps, 2))
        for t_idx, t in np.ndenumerate(tlist):
            # t = t_idx * dt
            A_list[t_idx] = get_A(t, x_traj[t_idx], u_traj[t_idx])
            B_list[t_idx] = get_B(t, x_traj[t_idx], u_traj[t_idx])
            a_list[t_idx] = dldx(t, x_traj[t_idx], u_traj[t_idx])
            b_list[t_idx] = dldu(t, x_traj[t_idx], u_traj[t_idx])

        xd_T = get_xd(tlist[-1])  # desired terminal state
        xT = x_traj[-1, :]

        plist = np.diag(P1)
        # p1 = np.zeros(3)  # replace it to be the terminal condition p(T)
        p1 = 2 * plist * (xT - xd_T)

        def zp_dyn(t, zp):
            zt = zp[:3]
            pt = zp[3:]

            t_idx = int(t / dt)
            At = A_list[t_idx]
            Bt = B_list[t_idx]
            at = a_list[t_idx]
            bt = b_list[t_idx]

            # M_11 = np.zeros((3,3))  # replace this
            M_11 = At
            M_12 = np.zeros((3, 3))  # replace this
            M_21 = np.zeros((3, 3))  # replace this
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
            zp_dot_list = np.zeros((6, list_len))
            for _i in range(list_len):
                zp_dot_list[:, _i] = zp_dyn(t_list[_i], zp_list[:, _i])
            return zp_dot_list

        # boundary condition (inputs are [z(0),p(0)] and [z(T),p(T)])
        def zp_bc(zp_0, zp_T):
            # return np.zeros(6)  # replace this
            z0 = zp_0[:3]
            p0 = zp_0[3:]

            zT = zp_T[:3]
            pT = zp_T[3:]

            bc = np.zeros(6)
            bc[:3] = z0
            bc[3:] = np.abs(pT - p1)
            # print(bc)

            return bc

        ### The solver will say it does not converge, but the returned result
        ### is numerically accurate enough for our use
        # zp_traj = np.zeros((tsteps,6))  # replace this by using solve_bvp
        res = solve_bvp(
            zp_dyn_list,
            zp_bc,
            tlist,
            np.zeros(shape=(6, tsteps)),
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

        z_traj = zp_traj[:, :3]
        p_traj = zp_traj[:, 3:]

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

        return v_traj

    # Start iLQR iterations here

    u_traj = init_u_traj.copy()
    Jlist = np.array([func_J(traj_sim(x0, u_traj), u_traj)])

    for _ in range(10):
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
        v_traj = ilqr_iter(x0, u_traj)
        # print(v_traj)

        # Armijo line search parameters
        gamma = 1.0  # initial step size
        alpha = 1e-04
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
        print(gamma)
        if gamma < 1e-5:
            break

    init_x_traj = traj_sim(x0, init_u_traj)
    xd_traj = np.zeros(shape=(tsteps, 3))
    for i in range(tsteps):
        xd_traj[i, :] = get_xd(tlist[i])

    # init_x_traj = np.vstack([x0, init_x_traj])
    # x_traj = np.vstack([x0, x_traj])
    # xd_traj = get_xd(tlist)

    print(x_traj.shape, u_traj.shape)
    # print(xd_traj)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].set_title("State Trajectory")
    axs[0].set_xlabel(r"$x(t)$")
    axs[0].set_ylabel(r"$y(t)$")
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
        label="Converged Trajectory",
    )
    axs[0].plot(
        xd_traj[:, 0],
        xd_traj[:, 1],
        linestyle="--",
        label="Desired Trajectory",
    )
    # axs[0].set_aspect("equal")
    axs[0].set_xlim(-0.3, 4.3)
    axs[0].set_ylim(-0.3, 4.3)
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
    # plt.close()

    plt.savefig(fname=os.path.join(RESULTS_DIR, fig_filename), format="png", dpi=300)


# xd_vec


if __name__ == "__main__":
    main_p2(n_iter=100, start=(-4, -2), alpha=1e-4, beta=0.5)

    dt = 0.1
    T = 2.0 * np.pi
    tlist = np.arange(0, T, dt)
    tsteps = tlist.shape[0]
    init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(tsteps, 1))
    Q_x = np.diag([10.0, 10.0, 2.0])
    R_u = np.diag([4.0, 2.0])
    P1 = np.diag([20.0, 20.0, 5.0])

    Q_z = np.diag([5.0, 5.0, 1.0])
    R_v = np.diag([2.0, 1.0])
    main_p3(
        dt=dt,
        T=T,
        init_u_traj=init_u_traj,
        Q_x=Q_x,
        R_u=R_u,
        P1=P1,
        Q_z=Q_z,
        R_v=R_v,
        fig_filename="part3_1.png",
    )

    dt = 0.1
    T = 2.0 * np.pi
    tlist = np.arange(0, T, dt)
    tsteps = tlist.shape[0]
    # init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(tsteps, 1))
    init_u_traj = np.vstack([np.cos(tlist), np.sin(tlist)]).T
    Q_x = np.diag([25.0, 30.0, 5.0])
    R_u = np.diag([12.0, 7.0])
    P1 = np.diag([20.0, 20.0, 5.0])

    Q_z = np.diag([7.0, 7.0, 2.5])
    R_v = np.diag([1.5, 1.1])
    main_p3(
        dt=dt,
        T=T,
        init_u_traj=init_u_traj,
        Q_x=Q_x,
        R_u=R_u,
        P1=P1,
        Q_z=Q_z,
        R_v=R_v,
        fig_filename="part3_2.png",
    )

    dt = 0.1
    T = 2.0 * np.pi
    tlist = np.arange(0, T, dt)
    tsteps = tlist.shape[0]
    # init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(tsteps, 1))
    init_u_traj = np.random.multivariate_normal(
        mean=np.array([1.0, -0.5]),
        cov=np.array([[1, 0], [0, 1]]),
        size=tsteps,
    )
    Q_x = np.diag([10.0, 10.0, 2.0])
    R_u = np.diag([4.0, 2.0])
    P1 = np.diag([20.0, 20.0, 5.0])

    Q_z = np.diag([5.0, 5.0, 1.0])
    R_v = np.diag([2.0, 1.0])
    main_p3(
        dt=dt,
        T=T,
        init_u_traj=init_u_traj,
        Q_x=Q_x,
        R_u=R_u,
        P1=P1,
        Q_z=Q_z,
        R_v=R_v,
        fig_filename="part3_3.png",
    )
