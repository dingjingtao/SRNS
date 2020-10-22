# SRNS

This repository is the official implementation of **Simplify and Robustify Negative Sampling for**
**Implicit Collaborative Filtering** (Accepted by NeurIPS 2020) 

## Requirements

We conducted experiments under：

- python 3.7
- tensorflow 1.14

## Training

#### Synthetic dataset

To tune the hyper-parameter on synthetic dataset（Ecom-toy） with uniform negative sampling method:

```shell
$ cd ./SRNS_syn
$ ./run_uniform.sh
```

To run the synthetic experiments on synthetic dataset:

```shell
$ cd ./SRNS_syn
$ ./run_srns.sh
```

For SRNS experiment, we constantly feed a false negative into each user’s memory $M$, you can tune the hyper-parameter $\sigma \in [1 ,3, 5, 7, 9]$ to vary the size of available false negative instance. The data of different size of available false negative instance are named `fn_sigma_*.pkl`, which means $\sigma \in [0.1, 0.3, 0.5, 0.7, 1.0]$  actually. They are under the `./SRNS_syn/toy` and `./SRNS_syn/toy_tuning` folder. 

**Notice**: hyper-parameter tuning and formal experiments use the different default dataset, they are all generated from the Ecom-toy dataset.  You can also run the synthetic experiments on the hyper-parameter tuning dataset. 

#### Real world dataset

To run SRNS on Ml-1m

```sh
$ cd ./SRNS_real
$ ./run.sh
```

## Pre-trained Models

You can get pre-trained models in `./SRNS_real/model`, it includes SRNS and all other baselines in our paper.

To evaluate the pre-trained models

```shell
$ cd ./SRNS_real
$ ./run_predict_model.sh
```

## Results

The performance of the pretraind model on ML-1M:

| Model   | N@1        | N@3        | R@3        |
| ------- | ---------- | ---------- | ---------- |
| ENMF    | 0.1846     | 0.2970     | 0.3804     |
| Uniform | 0.1744     | 0.2846     | 0.3663     |
| NNCF    | 0.0831     | 0.1428     | 0.1873     |
| AOBPR   | 0.1782     | 0.2907     | 0.3749     |
| IRGAN   | 0.1763     | 0.2878     | 0.3706     |
| RNS-AS  | 0.1810     | 0.2950     | 0.3801     |
| AdvIR   | 0.1792     | 0.2889     | 0.3699     |
| SRNS    | **0.1911** | **0.3056** | **0.3907** |

