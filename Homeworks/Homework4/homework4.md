---
geometry: margin=2cm
---

# ME455 Active Learning for Robotics
**Author**: Jingkun (Allen) Liu

## Problem 1
The ODE is shown below:

$$
\begin{aligned}
    \dot{p}(t) &= -A(t)^\top p(t) -a_z(t) \\
    \dot{z}(t) &= A(t)z(t) + B(t)v(t) 
\end{aligned}
$$

So that is can be expressed as 

$$
\begin{aligned}
    \begin{bmatrix}
        \dot{p}(t) \\
        \dot{z}(t)
    \end{bmatrix} &= \begin{bmatrix}
        -A(t)^\top & 0\\
        0 & A(t)
    \end{bmatrix} \begin{bmatrix}
        p(t) \\
        z(t)
    \end{bmatrix} + \begin{bmatrix}
        -a_z(t) \\
        B(t)v(t)
    \end{bmatrix}
\end{aligned}
$$

where $a_z(t)$ and $b_v(t)$ are 

$$
\begin{aligned}
    a_z(t) &= \\
    b_v(t) &=
\end{aligned}
$$