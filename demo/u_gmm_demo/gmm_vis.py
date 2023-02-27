import matplotlib.pyplot as plt
import numpy as np

from ml_cur.distribution.mixture import GMM


class Colors:
    """Provides colors for plotting"""

    def __init__(self, pyplot_color_cycle=True):
        if pyplot_color_cycle:
            self._colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        else:
            raise NotImplementedError("Not yet implemented")

    def __call__(self, i):
        return self._colors[i % len(self._colors)]


class Gmm2dVisualizer:
    def __init__(self):
        self._colors = Colors()

    def plot_model(
        self,
        _model,
        ax: plt.Axes = None,
        true_model=None,
        data=None,
        y_label=False,
        title: str = None,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        line_list = []
        label_list = []
        if data is not None:
            ax.scatter(x=data[:, 0], y=data[:, 1], alpha=0.1, c="tab:gray")

        if true_model is not None:
            for i, c in enumerate(true_model.components):
                (l,) = self._draw_2d_covariance(
                    ax,
                    c.mean,
                    c.covar,
                    c=self._colors(0),
                    linestyle=(0, (3, 1, 1, 1, 1, 1)),
                )

        if _model is not None:
            model = _model
            if isinstance(model, GMM):
                for i, c in enumerate(model.components):
                    (l,) = self._draw_2d_covariance(
                        ax, c.mean, c.covar, c=self._colors(i + 1)
                    )
                    line_list.append(l)
                    label_list.append(
                        np.array(model.weight_distribution.probabilities())[i]
                    )
            else:
                for i in range(model.num_components):
                    mean, covar = (
                        model.components[i].mean(),
                        model.components[i].covar(),
                    )
                    (l,) = self._draw_2d_covariance(
                        ax, mean, covar, c=self._colors(i + 1)
                    )
                    line_list.append(l)
                    label_list.append(np.array(model.weight_distribution.probs[i]))
        if len(line_list) > 0:
            ax.legend(line_list, ["{:.3f}".format(x) for x in label_list])

        ax.grid(True, linestyle="dotted")
        if y_label:
            ax.set_ylabel("Y Position")
        ax.set_xlabel("X Position")

        if title is not None:
            ax.set_title(title)
        return l

    def _draw_2d_covariance(
        self,
        ax,
        mean,
        covmatrix,
        chisquare_val=2.4477,
        return_raw=False,
        *args,
        **kwargs
    ):
        (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(covmatrix)
        phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

        a = chisquare_val * np.sqrt(largest_eigval)
        b = chisquare_val * np.sqrt(smallest_eigval)

        ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi))
        ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi))

        R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
        if return_raw:
            return mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1]
        else:
            return ax.plot(
                mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1], *args, **kwargs
            )
