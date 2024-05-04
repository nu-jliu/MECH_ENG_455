import copy
import os
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


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


if __name__ == "__main__":
    n_iter = 100
    x_range = np.linspace(-5, 5, 1000)
    y_range = np.linspace(-5, 5, 1000)

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = opt_func(x1=x_grid, x2=y_grid)

    gamma_0 = 1
    alpa = 1e-4
    beta = 0.5
    x0 = np.array([-4, -2], dtype="float")
    x = copy.deepcopy(x0)

    traj = np.zeros(shape=(n_iter + 1, 2))
    traj[0, :] = x

    for i in range(n_iter):
        gamma = gamma_0
        z = -gradient(*x)

        while (opt_func(*(x + gamma * z))) > opt_func(*x) + alpa * gamma * np.dot(
            gradient(*x), z
        ):
            gamma *= beta

        x += gamma * z
        traj[i + 1, :] = x

    # print(traj)

    # print(z_grid)
    plt.imshow(
        z_grid,
        origin="lower",
        extent=[-5, 5, -5, 5],
        cmap="Blues_r",
        norm=mcolors.LogNorm(),
    )
    cbar = plt.colorbar()
    # plt.contourf(x_grid, y_grid, z_grid, cmap="Blues_r", norm=mcolors.LogNorm())
    plt.plot(traj[:, 0], traj[:, 1], color="m", linewidth=1, label="Iterations")
    plt.scatter(*traj[-1, :], color="m", label="Converged estimation")
    plt.scatter(*x0, color="k", label="Initial guess")
    plt.legend()
    # cbar.set_ticks(
    #     [
    #         10**i
    #         for i in range(
    #             int(np.floor(np.log10(np.min(z_grid)))),
    #             int(np.ceil(np.log10(np.max(z_grid)))) + 1,
    #         )
    #     ]
    # )
    # plt.show()
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
