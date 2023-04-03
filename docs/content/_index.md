###
{{< doublevideo src1="img/obstacle_avoidance.webm" src2="img/table_tennis.webm">}}
# Abstract
Learning skills by imitation is a promising concept for the intuitive teaching of robots. A common way to learn such skills is to learn a parametric model by maximizing the likelihood given the demonstrations. 

Yet, human demonstrations are often multi-modal, i.e., the same task is solved in multiple ways which is a major challenge for most imitation learning methods that are based on such a maximum likelihood (ML) objective. The ML objective forces the model to cover all data, it prevents specialization in the context space and can cause mode-averaging in the behavior space, leading to suboptimal or potentially catastrophic behavior. 

Here, we alleviate those issues by introducing a curriculum using a weight for each data point, allowing the model to specialize on data it can represent while incentivizing it to cover as much data as possible by an entropy bonus. We extend our algorithm to a Mixture of (linear) Experts (MoE) such that the single components can specialize on local context regions, while the MoE covers all data points. 

We evaluate our approach in complex simulated and real robot control tasks and show it learns from versatile human demonstrations and significantly outperforms current SOTA methods.

## Installation
Clone our [Github repository](https://github.com/intuitive-robots/ml-cur) and install with `pip`:
```sh
git clone https://github.com/intuitive-robots/ml-cur.git
pip install ml-cur
```

## Usage
Our public interfaces are inspired by `scikit-learn`. You can also find some Jupyter notebooks in [our demo folder](https://github.com/intuitive-robots/ml-cur/tree/main/demo).

```python {linenos=true}
from ml_cur import MlCurLinMoe
ml_cur_moe = MlCurLinMoe(n_components=2, train_iter=50, num_active_samples=0.4)
ml_cur_moe.fit(train_samples, train_contexts)
```

## Citation
If you find our work useful, please consider citing:
```BibTeX
@Article{Li2023Curriculum,
  author = {Li, Maximilian Xiling 
            and Celik, Onur 
            and Becker, Philipp 
            and Blessing, Denis 
            and Lioutikov, Rudolf 
            and Neumann, Gerhard},
  title  = {Curriculum-Based Imitation of Versatile Skills},
  year   = {2023},
}
```