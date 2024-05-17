import sys

import numpy as np
import scipy.stats

import matplotlib as mplt

import question1 as q1
import question2 as q2
import question3 as q3

# import IPython

# IPython.embed()

# from q1new import *


def init_param():
    mplt.rcParams["axes.linewidth"] = 3
    mplt.rcParams["axes.titlesize"] = 20
    mplt.rcParams["axes.labelsize"] = 20
    mplt.rcParams["axes.titlepad"] = 8.0
    mplt.rcParams["xtick.major.size"] = 6
    mplt.rcParams["xtick.major.width"] = 3
    mplt.rcParams["xtick.labelsize"] = 20
    mplt.rcParams["ytick.major.size"] = 6
    mplt.rcParams["ytick.major.width"] = 3
    mplt.rcParams["ytick.labelsize"] = 20
    mplt.rcParams["lines.markersize"] = 5
    mplt.rcParams["legend.fontsize"] = 15


def main():
    # init_param()
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python homework5.py <problem num>")

    num_p = int(sys.argv[1])

    w1, w2, w3 = 0.5, 0.2, 0.3

    mu1 = np.array([0.35, 0.38])
    mu2 = np.array([0.68, 0.25])
    mu3 = np.array([0.56, 0.64])

    cov1 = np.array([[1e-2, 4e-3], [4e-3, 1e-2]])
    cov2 = np.array([[5e-3, -3e-3], [-3e-3, 5e-3]])
    cov3 = np.array([[8e-3, 0.0], [0.0, 4e-3]])

    T = 10
    dt = 0.1
    tlist = np.arange(0, T, dt)
    tsteps = tlist.shape[0]

    u_traj = np.zeros(shape=(100, 2))
    x_curr = np.array([0.3, 0.3])

    weights = np.array([w1, w2, w3])
    means = np.array([mu1, mu2, mu3])
    covs = np.array([cov1, cov2, cov3])

    # N = N_grid + 1

    # dx = dy = 1.0 / N_grid

    dists = []
    for mu, cov in zip(means, covs):
        dist = scipy.stats.multivariate_normal(mean=mu, cov=cov)
        dists.append(dist)

    for i in range(100):
        x_new = q1.dyn_random(None, weights, dists)
        u_traj[i, :] = (x_new - x_curr) / dt
        x_curr = x_new
    # u_traj = (
    #     -0.07
    #     * np.array(
    #         [
    #             np.sin(tlist * np.pi / T + np.pi / 4),
    #             np.cos(tlist * np.pi / T - np.pi / 4),
    #         ]
    #     ).T
    # )
    # u_traj = 0.3 * np.array([np.sin(tlist), np.cos(tlist)]).T
    # if num_p == 1:
    #     u_traj = np.tile(np.array([0.05, 0.001]), reps=(tsteps, 1))

    # elif num_p == 2:
    #     u_traj = np.tile(np.array([0.005, 0.0015]), reps=(tsteps, 1))
    # u_traj = np.vstack(
    #     (
    #         np.tile(np.array([0.08, -0.02]), reps=(50, 1)),
    #         np.tile(np.array([-0.02, 0.08]), reps=(50, 1)),
    #     )
    # )
    # u_traj = np.random.uniform(low=-0.5, high=0.5, size=(100, 2))
    # u_traj = np.random.multivariate_normal(
    #     mean=np.array([0, 0]), cov=np.eye(2) * (0.5 / 3) ** 2, size=100
    # )
    # print(u_traj)

    if num_p == 1:
        u_traj = np.tile(np.array([0.05, 0.02]), reps=(tsteps, 1))

        q1.main_p1(
            w1=w1,
            w2=w2,
            w3=w3,
            mu1=mu1,
            mu2=mu2,
            mu3=mu3,
            cov1=cov1,
            cov2=cov2,
            cov3=cov3,
            init_u_traj=u_traj,
            q=0.1,
            R_u=np.diag([0.001, 0.001]),
            P1=np.diag([2.0, 2.0]),
            Q_z=np.diag([0.01, 0.001]),
            R_v=np.diag([0.001, 0.001]),
            x0=np.array([0.3, 0.3]),
            # u_traj=u_traj,
            N_grid=100,
            K_per_dim=10,
            dt=0.1,
            T=T,
            max_iter=100,
            gamma_0=0.001,
            fig_filename="problem1.png",
        )

    elif num_p == 2:
        u_traj = np.tile(np.array([0.005, 0.0015]), reps=(tsteps, 1))

        q2.main_p2(
            w1=w1,
            w2=w2,
            w3=w3,
            mu1=mu1,
            mu2=mu2,
            mu3=mu3,
            cov1=cov1,
            cov2=cov2,
            cov3=cov3,
            init_u_traj=u_traj,
            q=0.1,
            R_u=np.diag([0.01, 0.01]),
            P1=np.diag([2.0, 2.0]),
            Q_z=np.diag([0.01, 0.01, 0.001, 0.001]),
            R_v=np.diag([0.01, 0.01]),
            x0=np.array([0.3, 0.3, 0.0, 0.0]),
            # u_traj=u_traj,
            N_grid=100,
            K_per_dim=10,
            dt=0.1,
            T=T,
            max_iter=100,
            gamma_0=0.001,
            fig_filename="problem2.png",
            multiplier=1,
        )

    elif num_p == 3:
        u_traj = np.tile(np.array([0.02 * np.pi, -np.pi / 10.0]), reps=(tsteps, 1))

        q3.main_p3(
            w1=w1,
            w2=w2,
            w3=w3,
            mu1=mu1,
            mu2=mu2,
            mu3=mu3,
            cov1=cov1,
            cov2=cov2,
            cov3=cov3,
            init_u_traj=u_traj,
            q=0.1,
            R_u=np.diag([0.01, 0.001]),
            P1=np.diag([2.0, 2.0]),
            Q_z=np.diag([0.01, 0.01, 0.001]),
            R_v=np.diag([0.01, 0.005]),
            x0=np.array([0.3, 0.3, np.pi / 2.0]),
            # u_traj=u_traj,
            N_grid=100,
            K_per_dim=10,
            dt=0.1,
            T=T,
            max_iter=100,
            gamma_0=0.001,
            fig_filename="problem3.png",
            multiplyer=2,
        )

    else:
        raise RuntimeError("Invalid argument")


if __name__ == "__main__":
    main()
    # IPython.embed()
