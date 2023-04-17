# ml-cur


<div align = center>

[![Button page]][Link page]
[![Button arxiv]][Link]
<!----------------------------------------------------------------------------->
[Link page]: https://intuitive-robots.github.io/ml-cur/
[Button page]: https://img.shields.io/badge/Page-informational?style=for-the-badge&logoColor=white&logo=github


<!----------------------------------------------------------------------------->
[Link]: https://arxiv.org/abs/2304.05171
[Button arxiv]: https://img.shields.io/badge/Arxiv-red?style=for-the-badge&logoColor=white&logo=adobeacrobatreader
</div>

### ML-Cur Implementation

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