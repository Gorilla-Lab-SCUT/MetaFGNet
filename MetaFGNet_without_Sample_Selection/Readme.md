## MetaFGNet without sample selection
We explain some expandable options in our released code:
* **--num-updates-for-gradient:** It is set to 1 in all our experiments, but other value can also be applied.
* **--meta-sgd:** It is proposed in: https://arxiv.org/abs/1707.09835. We set this option to False in all our experiments.
* **--second-order-grad:** The first-order approximation and the full second-order implementation are proposed in: https://arxiv.org/abs/1703.03400, which shows both the two implementations give similar results.
* **--first-meta-update:** It is explicitly proposed in: https://arxiv.org/abs/1710.03463 and it can be viewed as a supplement to the original meta-learning loss. We set it to False in our experiments.
  
We provide those options for further extention.
