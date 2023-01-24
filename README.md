# Instructions

## Pre-requisites

Set up a python3.8 virtual environment using ```requirements.txt```.
Please set up a WandB account and project to be used for the logging.

For reproducible model forwards, set environment variable
```CUBLAS_WORKSPACE_CONFIG=:4096:8:```

## Running experiments

We supply a numerically stabilised python/numpy implementation of MEC that has quadratic runtime with
maximum dimension of both p and q.
Please note that, for the results quoted in our paper, we use a highly optimised C++ implementation,
so speed results from this code may not be comparable for iMEC. 

We will release our optimised implementations at a later stage, please contact cs@robots.ox.ac.uk for further information.

### Running iMEC experiments

For running iMEC at block size 10 on GPT-2 (90%) use the following:

```
python3 main.py 
--block-size 10
--group-name <WANDB_GROUP_NAME>
--method imec 
--medium meteor
--medium-entropy-loss-threshold 0.9
--medium-top-k 0 
--message-mode randombits
--model-device cuda:0
--name <WANDB_EXPERIMENT_NAME>
--stop-after-n-trajectories 100
--wandb-project <WANDB_PROJECT>
--wandb-entity <WANDB_ENTITY>
```

For GPT-2 (topk 40) set
```
--medium-entropy-loss-threshold 0.0
--medium-top-k 40
```

For WaveRNN set
```
--medium wavernn
```

For UNIF set
```
--medium random
--medium-top-k 40
--medium-entropy-loss-threshold 0.0
```

### Running Meteor experiments

For running Meteor at precision 32 on GPT-2 (90%) use the following:
```
python3 main.py 
--group-name <WANDB_GROUP_NAME>
--meteor-is-sort 1
--meteor-precision 32
--method meteor 
--medium meteor
--medium-entropy-loss-threshold 0.9
--medium-top-k 0 
--message-mode randombits
--model-device cuda:0
--name <WANDB_EXPERIMENT_NAME>
--stop-after-n-trajectories 100
--wandb-project <WANDB_PROJECT>
--wandb-entity <WANDB_ENTITY>
```

To instead run Meteor:reorder, set
```
--meteor-is-sort 1
```

### Running Arithmetic coding experiments 

For running Arithmetic coding at precision 16 on GPT-2 (90%) use the following:
```
python3 main.py 
--group-name <WANDB_GROUP_NAME>
--meteor-is-sort 1
--meteor-precision 16
--method arithmetic
--medium meteor
--medium-entropy-loss-threshold 0.9
--medium-top-k 0 
--message-mode randombits
--model-device cuda:0
--name <WANDB_EXPERIMENT_NAME>
--stop-after-n-trajectories 100
--wandb-project <WANDB_PROJECT>
--wandb-entity <WANDB_ENTITY>
```
