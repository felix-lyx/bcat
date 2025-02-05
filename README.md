# BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics

This repository contains code for the paper [BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://www.arxiv.org/abs/2501.18972).

The code is based on the repository [PROSE](https://github.com/felix-lyx/prose).

## Install dependencies

Using conda and the ```env.yml``` file:

```
conda env create --file=env.yml
```

## Run the model

To launch a model training with modified arguments (arg1,val1), (arg2,val2):

```
python main.py arg1=val1 arg2=val2
```

All default arguments can be found in the ```src/configs``` folder, and are managed using [Hydra](https://hydra.cc/).

Scripts for reproducing the results in the paper are located in `src/scripts` folder. 

## Data

The dataset we used are collected from [PDEBench](https://github.com/pdebench/PDEBench), [PDEArena](https://github.com/pdearena/pdearena), and [CFDBench](https://github.com/luo-yining/CFDBench). More details about data preprocessing are included in ```src/data_utils/README.md```.


## Distributed training

Distributed training is available via PyTorch Distributed Data Parallel (DDP)

To launch a run on 1 node with 4 GPU, use 

```
torchrun --standalone --nnodes 1 --nproc_per_node 4 main.py
```

## Citation

If you find our paper and code useful, please consider citing:

[BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://www.arxiv.org/abs/2501.18972)

```
@article{liu2025bcat,
  title={{BCAT}: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics},
  author={Yuxuan Liu and Jingmin Sun and Hayden Schaeffer},
  journal={arXiv preprint arXiv:2501.18972},
  year={2025}
}
```
