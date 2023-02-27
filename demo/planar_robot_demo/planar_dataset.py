import os

import numpy as np


def planar_train_test():
    """

    Returns:
        tuple: train_contexts, train_params, test_contexts, test_params
    """
    d = dict(
        np.load(
            os.path.join(os.path.dirname(__file__), "planar_data.npz"),
            allow_pickle=True,
        )
    )

    return d["train_contexts"], d["train_params"], d["test_contexts"], d["test_params"]
