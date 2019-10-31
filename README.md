### PBT: Population Based Training

[Population Based Training of Neural Networks, Jaderberg et al. @ DeepMind](https://arxiv.org/abs/1711.09846)

A implementation of PBT in PyTorch.

### Objective

Finding a good hyper-parameter schedule for neural networks.

### How does PBT work?
PBT is a novel, Lamarckian evolutionary approach to hyper-parameter optimization for selecting the optimal hyper-parameter configuration and machine learning model by training a series of neural network models in parallel. The method can be performed as quickly as other methods and has shown to outperform random search in model performance on various benchmarks in deep reinforcement learning using A3Cstyle methods, as well as in supervised learning for machine translation and Generative Adversarial Networks (GANs). While similar procedures have been explored independently, PBT has gained increasing amount of attention since it was proposed. There has already been seen shown various use cases of PBT in AutoML, e.g. packages for HPO tuning and frameworks. PBT have also streamlined the experiment testing in different application-based domains with different machine learning approaches such as auto-encoders, reinforcement learners, neural networks and generative adversarial networks.

In order to approach the problem, PBT considers a population consisting of N members, initially formed with different hyper-parameters sampled from a uniform distribution. The goal is to determine the optimal model across the population, which PBT achieves by adapting the hyper-parameters and copying weights based on some criteria. The approach defines two distinct methods, exploit and explore, that influence the hyper-prameters and the weights. In short, the exploit method decides whether the member should continue exploring the current solution or simply abandon it. If exploration is considered, the explore method is used to provide a new set of hyper-parameters $\lambda$. The initiative for exploitation or exploration depends on the individual performances of the entire population where the worst performing members exploits the best performing members, and the best performing members continue exploring.

For more information, see [the paper](https://arxiv.org/abs/1711.09846) or [blog post](https://deepmind.com/blog/population-based-training-neural-networks/).
