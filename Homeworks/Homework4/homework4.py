import os
import sys
import numpy as np

from question2 import *
from question3 import *

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


if __name__ == "__main__":
    # print(s)
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python homework4.py <question num>")

    q_num = int(sys.argv[1])

    if q_num == 2:
        main_p2(n_iter=100, start=(-4, -2), alpha=1e-4, beta=0.5)

    elif q_num == 3:
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

    else:
        raise RuntimeError("Invalid argument")
