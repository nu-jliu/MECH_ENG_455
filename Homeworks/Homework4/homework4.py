import copy
import os
import numpy as np
from scipy.integrate import solve_bvp
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
    u1 = ut[1]
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
    xd = np.array([4.0 / (2.0 * np.pi) * t, 0.0, np.pi / 2.0])
    return xd


def step(xt, ut, dt):
    # xt_new = xt + dt * dyn(xt, ut)  # recommended: replace it with RK4 integration
    k1 = dt * dyn(xt, ut)
    k2 = dt * dyn(xt + k1 / 2.0, ut)
    k3 = dt * dyn(xt + k2 / 2.0, ut)
    k4 = dt * dyn(xt + k3, ut)

    xt_new = xt + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return xt_new


def traj_sim(x0, ulist, dt):
    tsteps = ulist.shape[0]
    x_traj = np.zeros((tsteps, 3))
    xt = copy.deepcopy(x0)
    for t in range(tsteps):
        xt_new = step(xt, ulist[t], dt)
        x_traj[t] = copy.deepcopy(xt_new)
        xt = copy.deepcopy(xt_new)
    return x_traj


def loss(t, xt, ut, Qx, Ru):
    # xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])  # desired system state at time t
    xd = get_xd(t)

    x_diff = xt - xd
    # x_loss = 0.0  # replace this
    x_loss = x_diff.T @ Qx @ x_diff
    # u_loss = 0.0  # replace this
    u_loss = ut.T @ Ru @ ut

    return x_loss + u_loss


def dldx(t, xt, ut, Qx, Ru):
    # xd = np.array([2.0 * t / np.pi, 0.0, np.pi / 2.0])
    xd = get_xd(t)
    qlist = np.diag(Qx)

    # dvec = np.zeros(3)  # replace this
    dx = xt - dx
    dvec = qlist * 2 * dx
    return dvec


def dldu(t, xt, ut, Qx, Ru):
    # dvec = np.zeros(2)  # replace this
    rlist = np.diag(Ru)
    dvec = 2 * rlist * ut
    return dvec


def ilqr_iter(x0, u_traj, tsteps, dt):
    """
    :param x0: initial state of the system
    :param u_traj: current estimation of the optimal control trajectory
    :return: the descent direction for the control
    """
    # forward simulate the state trajectory
    x_traj = traj_sim(x0, u_traj, dt)

    # compute other variables needed for specifying the dynamics of z(t) and p(t)
    A_list = np.zeros((tsteps, 3, 3))
    B_list = np.zeros((tsteps, 3, 2))
    a_list = np.zeros((tsteps, 3))
    b_list = np.zeros((tsteps, 2))
    for t_idx in range(tsteps):
        t = t_idx * dt
        A_list[t_idx] = get_A(t, x_traj[t_idx], u_traj[t_idx])
        B_list[t_idx] = get_B(t, x_traj[t_idx], u_traj[t_idx])
        a_list[t_idx] = dldx(t, x_traj[t_idx], u_traj[t_idx])
        b_list[t_idx] = dldu(t, x_traj[t_idx], u_traj[t_idx])

    xd_T = np.array(
        [2.0 * (tsteps - 1) * dt / np.pi, 0.0, np.pi / 2.0]
    )  # desired terminal state
    p1 = np.zeros(3)  # replace it to be the terminal condition p(T)

    def zp_dyn(t, zp):
        t_idx = (t / dt).astype(int)
        At = A_list[t_idx]
        Bt = B_list[t_idx]
        at = a_list[t_idx]
        bt = b_list[t_idx]

        M_11 = np.zeros((3, 3))  # replace this
        M_12 = np.zeros((3, 3))  # replace this
        M_21 = np.zeros((3, 3))  # replace this
        M_22 = np.zeros((3, 3))  # replace this
        dyn_mat = np.block([[M_11, M_12], [M_21, M_22]])

        m_1 = np.zeros(3)  # replace this
        m_2 = np.zeros(3)  # replace this
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
        return np.zeros(6)  # replace this

    ### The solver will say it does not converge, but the returned result
    ### is numerically accurate enough for our use
    zp_traj = np.zeros((tsteps, 6))  # replace this by using solve_bvp

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

        vt = np.zeros(2)  # replace this
        v_traj[_i] = vt

    return v_traj


def get_J(x_traj, u_traj, dt, T, P1):
    t = 0
    l_total = 0
    for xt, ut in zip(x_traj, u_traj):
        t += dt
        l_curr = loss(t, xt, ut)
        l_total += l_curr

    xT = x_traj[-1, :]
    xdT = get_xd(T)

    xdiff_T = xT - xdT
    mxT = xdiff_T.reshape(-1, 1) @ P1 @ xdiff_T

    l_total += mxT

    return l_total


def main_p3():
    ### define parameters

    dt = 0.1
    T = 2.0 * np.pi
    tlist = np.arange(0, T, dt)
    x0 = np.array([0.0, 0.0, np.pi / 2.0])
    tsteps = len(tlist)
    init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(tsteps, 1))

    Q_x = np.diag([10.0, 10.0, 2.0])
    R_u = np.diag([4.0, 2.0])
    P1 = np.diag([20.0, 20.0, 5.0])

    Q_z = np.diag([5.0, 5.0, 1.0])
    R_v = np.diag([2.0, 1.0])

    T = dt * tsteps

    # Start iLQR iterations here

    u_traj = copy.deepcopy(init_u_traj)
    for iter in range(10):
        # forward simulate the current trajectory
        x_traj = traj_sim(x0, u_traj)

        # visualize the current trajectory
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Iter: {:d}".format(iter))
        ax.set_aspect("equal")
        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(-0.2, 2.2)
        ax.plot(x_traj[:, 0], x_traj[:, 1], linestyle="-", color="C0")
        plt.show()
        plt.close()

        # get descent direction
        v_traj = ilqr_iter(x0, u_traj, tsteps, dt)

        # Armijo line search parameters
        gamma = 1.0  # initial step size
        alpha = 1e-04
        beta = 0.5

        ### Implement Armijo line search here to update step size gamma
        while get_J(x_traj, u_traj + gamma * v_traj, dt, T, P1) > get_J(
            x_traj, u_traj, dt, T, P1
        ) + alpha * gamma * np.dot(-v_traj, v_traj):
            gamma *= beta

        # update control for the next iteration
        u_traj += gamma * v_traj


if __name__ == "__main__":
    main_p2(n_iter=100, start=(-4, -2), alpha=1e-4, beta=0.5)
