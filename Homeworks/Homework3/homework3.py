import os
import sys
import requests
import copy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import cupy as cp
import scipy.stats

from typing import Literal
from PIL import Image
from io import BytesIO

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

IMAGE_URL = (
    "https://raw.githubusercontent.com/MurpheyLab/ME455_public/main/figs/lincoln.jpg"
)

#################### PROBLEM 1 ####################


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

    plt.savefig(f"{RESULTS_DIR}/part1_{dist}.png", format="png", dpi=300)
    print("Done sampling, plot saved")


#################### PROBLEM 2 ####################


def normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def func_qdot(state, u):
    theta = state[2]

    u1 = u[0]
    u2 = u[1]

    xdot = u1 * np.cos(theta)
    ydot = u1 * np.sin(theta)
    thetadot = u2

    qdot = np.array([xdot, ydot, thetadot])
    # qdot = np.random.multivariate_normal(qdot, np.eye(3) * noice)
    return qdot


def new_state(state, u, dt, noice=0, control_noice=0):
    u_noice = np.random.multivariate_normal(mean=u, cov=np.eye(2) * control_noice)

    st = copy.deepcopy(state)
    s1 = dt * func_qdot(st, u_noice)
    s2 = dt * func_qdot(st + s1 / 2, u_noice)
    s3 = dt * func_qdot(st + s2 / 2, u_noice)
    s4 = dt * func_qdot(st + s3, u_noice)

    state_new = st + 1.0 / 6.0 * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
    state_new[:3] = np.random.multivariate_normal(
        mean=state_new[:3], cov=np.eye(3) * noice
    )
    state_new[2] = normalize(state_new[2])
    return state_new


def get_measurement(sample, noice=0):
    mean = sample[:3]
    cov = np.eye(3) * noice
    return np.random.multivariate_normal(mean=mean, cov=cov)


def prob_density(z_vec, sample, noice):
    dist = scipy.stats.multivariate_normal(mean=sample[:3], cov=np.eye(3) * noice)
    density = dist.pdf(x=z_vec)

    return density


def update_plot(
    frame,
    ax,
    traj,
    traj_est,
    particles,
    colors,
):
    ax.cla()
    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(-1.0, 3.0)

    (line,) = ax.plot(
        traj[0, : (frame + 1)],
        traj[1, : (frame + 1)],
        linewidth=1,
        color="k",
    )

    ax.plot(
        traj_est[0, : (frame + 1)],
        traj_est[1, : (frame + 1)],
        linewidth=1,
        color="r",
    )

    for i in range(frame + 1):
        ax.scatter(
            particles[0, :, i].flatten(),
            particles[1, :, i].flatten(),
            color=colors[i],
            alpha=particles[3, :, i].flatten() / np.max(particles[3, :, i]),
            # edgecolors="k",
            s=20,
        )

    return (line,)


def p2_main(
    start: tuple[float, float, float] = (0.0, 0.0, np.pi / 2.0),
    u: tuple[float, float] = (1.0, -1.0 / 2.0),
    T: float = 2.0 * np.pi,
    dt: float = 0.1,
    noice_process: float = 0.002,
    noice_measure: float = 0.02,
    num_particles: int = 10,
):

    t_list = np.arange(start=0.0, stop=T, step=dt)
    num_steps = len(t_list)

    traj = np.zeros(shape=(3, num_steps))
    traj_est = np.zeros(shape=(3, num_steps))

    particles = np.zeros(shape=(4, num_particles, num_steps))

    Sigma_mat = np.eye(3) * noice_measure
    part_curr = np.random.multivariate_normal(
        mean=start, cov=Sigma_mat, size=num_particles
    )
    part_curr = np.c_[part_curr, np.ones(shape=(num_particles, 1)) * 1 / num_particles]

    state = copy.deepcopy(start)
    traj[:, 0] = state
    traj_est[:, 0] = state
    particles[:, :, 0] = part_curr.T
    sum_cum = 0

    indices = np.arange(0, num_particles)

    print("Start simulating ...")
    for i in range(1, num_steps):

        # state = new_state(state=state, u=u, dt=dt, noice=0)
        state = new_state(state=state, u=u, dt=dt, noice=noice_process)
        # state = new_state(state=state, u=u, dt=dt, control_noice=noice)
        z_vec = get_measurement(sample=state, noice=noice_measure)

        for j in range(num_particles):
            s = part_curr[j, :]
            # z_hat = get_measurement(sample=s)
            w = prob_density(
                z_vec=z_vec,
                sample=s,
                noice=noice_measure,
            )

            s_new = new_state(state=s[:3].T, u=u, dt=dt, noice=noice_process)
            part_curr[j, :3] = s_new
            part_curr[j, 3] *= w

        sum_weight = np.sum(part_curr[:, 3])
        part_curr[:, 3] /= sum_weight

        part_new = np.zeros_like(part_curr)
        ind_new = np.random.choice(a=indices, size=num_particles, p=part_curr[:, 3])

        # print(ind_new)
        for j in range(num_particles):
            ind = ind_new[j]
            part_new[j, :] = part_curr[ind, :]

        state_est = np.mean(part_new[:, :3], axis=0)
        part_curr = part_new

        sum_cum += sum_weight

        particles[:, :, i] = part_curr.T
        traj_est[:, i] = state_est
        traj[:, i] = state

    print("Plotting results ...")

    cmap_rainbow = plt.get_cmap("rainbow")
    frames = np.arange(0, num_steps)
    colors = cmap_rainbow(np.linspace(0, 1, num_steps))
    colors = [tuple(row) for row in colors]
    plot_steps = np.round(
        np.arange(start=num_steps - (6 / dt) - 1, stop=num_steps, step=1 / dt)
    )
    # plot_steps = np.arange(start=10, stop=num_steps, step=10)
    # plt.xlim(left=-1.0, right=5.0)
    # plt.ylim(bottom=-1.0, top=3.0)

    print(plot_steps)

    plt.plot(
        traj[0, :],
        traj[1, :],
        linewidth=1,
        color="k",
        label="Ground Truth",
    )
    plt.plot(
        traj_est[0, :],
        traj_est[1, :],
        linewidth=1,
        color="r",
        label="Estimated Trajectory",
    )

    for i in plot_steps:
        # for i in range(num_steps):
        ind = int(i)
        plt.scatter(
            particles[0, :, ind].flatten(),
            particles[1, :, ind].flatten(),
            color=colors[ind],
            # edgecolors="k",
            alpha=particles[3, :, ind].flatten() / np.max(particles[3, :, ind]),
            s=20,
        )

    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/part2.png", format="png", dpi=300)

    fig, ax = plt.subplots()
    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(-1.0, 3.0)

    animate = FuncAnimation(
        fig=fig,
        func=update_plot,
        fargs=(ax, traj, traj_est, particles, colors),
        # fargs=(line, scatter, traj, particles, noice, colors),
        frames=frames,
        interval=dt * 1e3,
        blit=True,
    )

    animate.save(f"{RESULTS_DIR}/part2.mp4", writer="ffmpeg")
    print("Simulation saved")


#################### PROBLEM 3 ####################


def generate_samples(weights, means, covs, num_samples):
    samples = np.zeros(shape=(num_samples, 3))

    indeces = np.arange(0, len(weights))

    for i in range(num_samples):
        index = np.random.choice(a=indeces, p=weights, size=1)[0]

        mean = means[index, :]
        cov = covs[index, :, :]

        s = np.random.multivariate_normal(mean=mean, cov=cov)
        samples[i, :2] = s
        samples[i, 2] = index

    return samples


def generate_pdf(weights, means, covs):
    grid_range = np.arange(0, 1, 0.01)
    x_grid, y_grid = np.meshgrid(grid_range, grid_range)
    pos = np.stack((x_grid, y_grid), axis=2)
    # print(pos.shape)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    len_grid = len(grid_range)
    pdf_gmm = np.zeros(shape=(len_grid, len_grid))

    for i in range(3):

        dist = scipy.stats.multivariate_normal(mean=means[i, :], cov=covs[i, :, :])
        pdf_curr = dist.pdf(pos)
        # print(np.max(pdf_curr))
        pdf_curr /= np.sum(pdf_curr)

        pdf_gmm += weights[i] * pdf_curr

    # print(pos)

    pdf_gmm /= np.sum(pdf_gmm)
    return pdf_gmm


def generate_boundary_points(mean, cov):
    eigval, eigvec = np.linalg.eig(cov)
    print(eigval)
    print(eigvec)
    thetalist = np.linspace(0, 2 * np.pi, 1000)
    eclipse = 30 * np.array(
        [
            eigval[0] * np.cos(thetalist),
            eigval[1] * np.sin(thetalist),
        ]
    )
    #
    # print(eclipse.T @ eigvec.T)
    points = mean + eclipse.T @ eigvec.T
    # print(points)
    return points


def p3_main(weights, means, covs, num_sample):

    if not weights.shape[0] == means.shape[0] == covs.shape[0]:
        raise RuntimeError("Invalid input configuration")

    num_dist = len(weights)

    fig, axs = plt.subplots(5, 3, figsize=(25, 30))

    for row in axs:
        for ax in row:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")

        samples = generate_samples(weights, means, covs, num_sample)

        pdf_gmm = generate_pdf(weights, means, covs)
        sample_dists = []

        for j in range(num_dist):
            mask = samples[:, 2] == j
            sample_dists.append(samples[mask])

        row[0].scatter(samples[:, 0], samples[:, 1])

        cax = row[1].imshow(
            pdf_gmm,
            cmap="gray_r",
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        fig.colorbar(cax, ax=row[1], orientation="vertical")

        for j in range(num_dist):
            row[1].scatter(sample_dists[j][:, 0], sample_dists[j][:, 1])

            mean = means[j, :]
            cov = covs[j, :]

            points = generate_boundary_points(mean, cov)
            # print(points.shape)
            row[1].plot(points[:, 0], points[:, 1])

    # axs[1].scatter(samples[mask_2][:, 0], samples[mask_2][:, 1])
    # axs[1].scatter(samples[mask_3][:, 0], samples[mask_3][:, 1])
    # plt.contourf(x_grid, y_grid, pdf_gmm, cmap="gray_r")
    plt.savefig(f"{RESULTS_DIR}/part3.png", format="png", dpi=300)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python homework3.py <Problem Number>")

    num_q = int(sys.argv[1])
    if num_q == 1:
        p1_main(dist="uniform", num_samples=5000)
        p1_main(dist="normal", num_samples=5000)

    elif num_q == 2:
        p2_main(
            start=(0, 0, np.pi / 2),
            u=(1, -1 / 2),
            T=2 * np.pi,
            dt=0.1,
            noice_process=0.002,
            noice_measure=0.02,
            num_particles=100,
        )

    elif num_q == 3:
        w1 = 0.5
        w2 = 0.2
        w3 = 0.3

        mean1 = np.array([0.35, 0.38])
        mean2 = np.array([0.68, 0.25])
        mean3 = np.array([0.56, 0.64])

        cov1 = np.array([[1e-2, 4e-3], [4e-3, 1e-2]])
        cov2 = np.array([[5e-3, -3e-3], [-3e-3, 5e-3]])
        cov3 = np.array([[8e-3, 0], [0, 4e-3]])

        weights = np.array([w1, w2, w3])
        means = np.array([mean1, mean2, mean3])
        covs = np.array([cov1, cov2, cov3])

        p3_main(weights=weights, means=means, covs=covs, num_sample=100)
