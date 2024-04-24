import matplotlib.lines
import numpy as np
import cupy as cp
import modern_robotics as mr

import requests
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import Literal
from PIL import Image
from io import BytesIO


IMAGE_URL = (
    "https://raw.githubusercontent.com/MurpheyLab/ME455_public/main/figs/lincoln.jpg"
)


def image_density(s, x_grid, y_grid, density_array):
    s_x, s_y = s

    ind_x = cp.argmin(cp.abs(x_grid - s_x))
    ind_y = cp.argmin(cp.abs(y_grid - s_y))

    val = density_array[ind_x, ind_y]

    return val


def p1_main(
    dist: Literal["uniform", "normal", "lognormal"] = "uniform",
    num_samples: int = 5000,
):
    print(f"Sampling {dist} distribution ...")

    response = requests.get(IMAGE_URL)
    image_data = BytesIO(response.content)

    image = Image.open(image_data)

    image_array = cp.array(image)
    image_array = cp.flipud(image_array)

    x_grids = cp.linspace(0.0, 1.0, image_array.shape[0])
    y_grids = cp.linspace(0.0, 1.0, image_array.shape[1])

    dx = x_grids[1]
    dy = y_grids[1]

    density_array = 255.0 - image_array
    density_array /= cp.sum(density_array) * dx * dy

    samples = None

    # Multivariable normal distribution
    if dist == "normal":
        var = ((0.5) / 3) ** 2
        samples = cp.random.multivariate_normal(
            mean=[0.5, 0.5], cov=[[var, 0], [0, var]], size=num_samples
        )

    # Uniform Distribution
    elif dist == "uniform":
        samples = cp.random.uniform(low=0.0, high=1.0, size=(num_samples, 2))

    elif dist == "log_normal":
        m_log = cp.log(0.5)
        s_log = cp.sqrt(cp.exp(var) * (cp.exp(var) - 1))
        samples = cp.random.lognormal(m_log, s_log, size=(num_samples, 2))

    else:
        raise RuntimeError("Invalid Distribution Type")

    sample_weights = cp.zeros(num_samples)

    for i in range(num_samples):
        s = samples[i]
        weight = image_density(s, x_grids, y_grids, density_array)
        sample_weights[i] = weight

    sample_weights /= cp.max(sample_weights)

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(
        cp.asnumpy(image_array), extent=(0, 1, 0, 1), origin="lower", cmap="gray"
    )
    axs[0].set_xlim(0.0, 1.0)
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_title("Probability Density Function")

    for s, w in zip(cp.asnumpy(samples), cp.asnumpy(sample_weights)):
        axs[1].plot(
            s[1], s[0], linestyle="", marker="o", markersize=2, color="k", alpha=w
        )

    axs[1].set_xlim(0.0, 1.0)
    axs[1].set_ylim(0.0, 1.0)
    axs[1].set_aspect("equal")
    axs[1].set_title("Samples")

    plt.savefig(f"part1_{dist}.png", format="png", dpi=300)
    print("Done sampling, plot saved")


def next_state(state: tuple[float, float, float], T):
    x = state[0]
    y = state[1]
    theta = state[2]

    Vs = np.array([0, 0, theta, x, y, 0])
    Tsb = mr.MatrixExp6(mr.VecTose3(Vs))
    Tsbp = Tsb @ T

    return get_state_from_tf(Tsbp)


def func_qdot(state, u):
    theta = state[2]

    u1 = u[0]
    u2 = u[1]

    xdot = u1 * np.cos(theta)
    ydot = u1 * np.sin(theta)
    thetadot = u2

    qdot = np.array([xdot, ydot, thetadot])
    return qdot


def new_state(state, u, dt):
    xt = copy.deepcopy(state)
    k1 = dt * func_qdot(xt, u)
    k2 = dt * func_qdot(xt + k1 / 2, u)
    k3 = dt * func_qdot(xt + k2 / 2, u)
    k4 = dt * func_qdot(xt + k3, u)

    state_new = xt + 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state_new


def get_state_from_tf(T):
    # v6_se3 = mr.MatrixLog6(T)
    # v6 = mr.se3ToVec(v6_se3)
    # x = v6[3]
    # y = v6[4]
    # theta = v6[2]

    R, p = mr.TransToRp(T)
    omg = mr.so3ToVec(mr.MatrixLog3(R))
    # theta = omg[2]
    # print(omg)
    tr = np.trace(R)
    theta = np.arccos((tr - 1) / 2)
    # theta = math.atan2(R[1][0], R[0][0])
    x = p[0]
    y = p[1]
    # print(v6)

    return np.array([x, y, theta])


def update_plot(
    frame,
    line: matplotlib.lines.Line2D,
    line_part: matplotlib.lines.Line2D,
    traj,
    particles,
    noice,
):
    line.set_xdata(traj[0, : (frame + 1)])
    line.set_ydata(traj[1, : (frame + 1)])

    # x, y, theta = traj[:, frame]

    # samples = np.random.multivariate_normal(
    #     mean=np.array([x, y, theta]),
    #     cov=np.eye(3) * noice,
    #     size=100,
    # )

    line_part.set_xdata(particles[0, :, : (frame + 1)].flatten())
    line_part.set_ydata(particles[1, :, : (frame + 1)].flatten())
    # print(particles[0, :, :1])
    return (line,)


def p2_main(
    start: tuple[float, float, float] = (0.0, 0.0, np.pi / 2.0),
    u: tuple[float, float] = (1.0, -1.0 / 2.0),
    T: float = 2.0 * np.pi,
    dt: float = 0.1,
    noice: float = 0.02,
):
    # t_list = np.linspace(start=0.0, stop=T, num=num_samples)
    t_list = np.arange(start=0.0, stop=T, step=dt)
    num_steps = len(t_list)
    num_samples = 100
    # dt = t_list[1]
    # start_V6 = np.array([0, 0, start[2], start[0], start[1], 0])
    # Tsb = mr.MatrixExp6(mr.VecTose3(start_V6))

    traj = np.zeros(shape=(3, num_steps))
    particles = np.zeros(shape=(3, num_samples, num_steps))
    particles[2, :, 0] = np.ones(shape=num_samples) * np.pi / 2
    # x, y, theta = get_state_from_tf(Tsb)
    state = copy.deepcopy(start)
    traj[:, 0] = state

    print(
        next_state(
            state,
            mr.MatrixExp6(mr.VecTose3(np.array([0, 0, -np.pi, 2 * np.pi, 0, 0]))),
        )
    )

    u1 = u[0]
    u2 = u[1]

    v3 = np.array([u2, u1, 0]) * dt
    v6 = np.r_[0, 0, v3, 0]
    Tbbp = mr.MatrixExp6(mr.VecTose3(v6))

    print("Start simulating ...")
    for i in range(1, num_steps):
        # Tsb = Tsb @ Tbbp
        # x, y, theta = get_state_from_tf(Tsb)
        # state = next_state(state=state, T=Tbbp)
        u_noice = np.random.multivariate_normal(
            mean=u,
            cov=np.eye(2) * noice,
            size=num_samples,
        )
        # print(u_noice)
        for j in range(num_samples):
            new_part = new_state(state=particles[:, j, i - 1], u=u_noice[j, :].T, dt=dt)
            particles[:, j, i] = new_part

        state = new_state(state=state, u=u, dt=dt)
        # state += np.random.normal(0, noice, 3)
        # traj[:, i + 1] = np.array([x, y, theta])
        traj[:, i] = state
        # print(state)

    print(traj)

    plt.plot(traj[0, :], traj[1, :])
    plt.xlim(left=-1.0, right=5.0)
    plt.ylim(bottom=-1.0, top=3.0)
    plt.savefig("part2.png", format="png", dpi=300)

    fig, ax = plt.subplots()
    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(-1.0, 3.0)
    (line,) = ax.plot(0, 0)
    (line_part,) = ax.plot(
        0,
        0,
        linestyle="",
        marker="o",
        markerfacecolor=(1, 0, 0, 0),
        markeredgecolor="k",
    )

    animate = FuncAnimation(
        fig=fig,
        func=update_plot,
        fargs=(line, line_part, traj, particles, noice),
        frames=np.arange(0, num_steps),
        interval=dt * 1e3,
        blit=True,
    )

    # print(particles)

    # plt.show()
    animate.save("part2.mp4", writer="ffmpeg")
    print("Simulation saved")


if __name__ == "__main__":
    # p1_main(dist="uniform", num_samples=5000)
    # p1_main(dist="normal", num_samples=5000)

    p2_main(
        start=(0, 0, np.pi / 2),
        u=(1, -1 / 2),
        T=2 * np.pi,
        dt=0.01,
        noice=0.02,
    )
