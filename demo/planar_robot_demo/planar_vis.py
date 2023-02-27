import matplotlib.pyplot as plt
import numpy as np


def fwd_kin(actions):
    angles_ = np.cumsum(actions, axis=1)
    pos0 = np.cos(angles_)
    pos1 = np.sin(angles_)
    return pos0, pos1


def draw_obstacles(ax):
    obstacles_x = [[2.5, 2.5, 3.3, 3.3], [2.7, 2.7, 3.5, 3.5], [3.0, 3.0, 3.8, 3.8]]
    obstacles_y = [
        [-0.5, 0.5, -0.5, 0.5],
        [-3.5, -2.5, -3.5, -2.5],
        [2.5, 3.5, 2.5, 3.5],
    ]
    for k in range(3):
        obstacle_x = obstacles_x[k]
        obstacle_y = obstacles_y[k]

        l1_x = [obstacle_x[0], obstacle_x[1]]
        l1_y = [obstacle_y[0], obstacle_y[1]]

        l2_x = [obstacle_x[1], obstacle_x[3]]
        l2_y = [obstacle_y[1], obstacle_y[3]]

        l3_x = [obstacle_x[0], obstacle_x[2]]
        l3_y = [obstacle_y[0], obstacle_y[2]]

        l4_x = [obstacle_x[2], obstacle_x[3]]
        l4_y = [obstacle_y[2], obstacle_y[3]]
        ax.plot(l1_x, l1_y, "r-", linewidth=2)
        ax.plot(l2_x, l2_y, "r-", linewidth=2)
        ax.plot(l3_x, l3_y, "r-", linewidth=2)
        ax.plot(l4_x, l4_y, "r-", linewidth=2)
    return ax


def visualize(actions, contexts=None, ax=None, alpha=0.4, title: str = None):
    if ax is None:
        fig, ax = plt.subplots()

    pos_x_in, pos_y_in = fwd_kin(actions)
    starting_point_x = 0
    starting_point_y = 0
    pos_x = np.cumsum(pos_x_in, axis=1)
    pos_y = np.cumsum(pos_y_in, axis=1)
    all_points_x = np.column_stack(
        (np.ones(pos_x.shape[0]) * starting_point_x, pos_x + starting_point_x)
    )
    all_points_y = np.column_stack(
        (np.ones(pos_y.shape[0]) * starting_point_y, pos_y + starting_point_y)
    )
    line_c = "grey"
    if alpha > 0.1:
        line_c = "black"

    for i in range(all_points_x.shape[0]):
        ax.plot(
            all_points_x[i, :],
            all_points_y[i, :],
            "o",
            color=line_c,
            alpha=alpha,
        )
        ax.plot(
            all_points_x[i, :],
            all_points_y[i, :],
            "-",
            color=line_c,
            linewidth=2,
            alpha=alpha,
        )
        ax.plot(
            all_points_x[i, -1],
            all_points_y[i, -1],
            "o",
            color="red",
            linewidth=2,
            alpha=alpha,
        )

    if contexts is not None:
        print(contexts[0])
        ax.plot(contexts[:, 0], contexts[:, 1], "bo", alpha=alpha)

    ax = draw_obstacles(ax)

    if title is not None:
        ax.set_title(title)
    return ax
