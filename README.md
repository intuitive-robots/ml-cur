# ml-cur
ML-Cur Implementation

This is our implementation for ML-Cur, a curriculum based approach for fitting distributions. It was introduced in our work **Curriculum-Based Imitation of Versatile Skills**.

## Installation
You can install our package by downloading this repository and calling:
`pip install <path-to-ml-cur>`

## Usage
Our Public Interfaces follow a structure inspired by `scikit-learn`. See also the IPython Notebooks in our [demo folder](demo/).

```python
from ml_cur import MlCurLinMoe
ml_cur_moe = MlCurLinMoe(n_components=2, train_iter=50, num_active_samples=0.4)
ml_cur_moe.fit(train_samples, train_contexts)
```


If you find our work useful, please consider citing:
```BibTeX
@INPROCEEDINGS{Li2023Curriculum,
  author = {Li, Maximilian Xiling 
            and Celik, Onur 
            and Becker, Philipp 
            and Blessing, Denis 
            and Lioutikov, Rudolf 
            and Neumann, Gerhard},
  title  = {Curriculum-Based Imitation of Versatile Skills},
  booktitle={2023 International Conference on Robotics and Automation (ICRA)},
  year   = {2023},
}
```