# Curriculum Based Imitation of Versatile Skills
{:refdef: style="text-align: center;"}
[:page_facing_up: PDF](/docs/Li2023CurriculumBasedImitation.pdf) [:computer: Code]([/docs/Li2023CurriculumBasedImitation.pdf](https://github.com/intuitive-robots/ml-cur))
{: refdef}

## Abstract
Learning skills by imitation is a promising concept for the intuitive teaching of robots. A common way to learn such skills is to learn a parametric model by maximizing the likelihood given the demonstrations. Yet, human demonstrations are often multi-modal, i.e., the same task is solved in multiple ways which is a major challenge for most imitation learning methods that are based on such a maximum likelihood (ML) objective. The ML objective forces the model to cover all data, it prevents specialization in the context space and can cause mode-averaging in the behavior space, leading to suboptimal or potentially catastrophic behavior. Here, we alleviate those issues by introducing a curriculum using a weight for each data point, allowing the model to specialize on data it can represent while incentivizing it to cover as much data as possible by an entropy bonus. We extend our algorithm to a Mixture of (linear) Experts (MoE) such that the single components can specialize on local context regions, while the MoE covers all data points. We evaluate our approach in complex simulated and real robot control tasks and show it learns from versatile human demonstrations and significantly outperforms current SOTA methods.

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
@Article{Li2023Curriculum,
  author = {Li, Maximilian Xiling and Celik, Onur and Becker, Philipp and Blessing, Denis and Lioutikov, Rudolf and Neumann, Gerhard},
  title  = {Curriculum-Based Imitation of Versatile Skills},
  year   = {2023},
}
```