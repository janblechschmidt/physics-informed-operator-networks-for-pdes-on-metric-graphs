# Physics-informed Operator Networks for PDEs on Metric Graphs

This repository accompanies the the ICML 2025 publication entitled **Physics-Informed DeepONets for drift-diffusion on metric graphs: simulation
and parameter identification**.

## Link to Paper

The paper is available here [https://openreview.net/forum?id=oF1OPyMw1m]

## Abstract

We develop a novel physics informed deep learning approach for solving nonlinear drift-diffusion equations on metric graphs.
These models represent an important model class with a large number of applications in areas ranging from transport in biological cells to the motion of human crowds.
While traditional numerical schemes require a large amount of tailoring, especially in the case of model design or parameter identification problems, physics informed deep operator networks (PI DeepONets) have emerged as a versatile tool for the solution of partial differential equations with the particular advantage that they easily incorporate parameter identification questions.
We here present an approach where we first learn three PI DeepONets models for representative inflow, inner and outflow edges, resp., and then subsequently couple these models for the solution of the drift-diffusion metric graph problem by relying on an edge-based domain decomposition approach.
We illustrate that our framework is applicable for the accurate evaluation of graph-coupled physics models and is well suited for solving optimization or inverse problems on these coupled networks.

## Citation

    @misc{blechschmidt2025physicsinformeddeeponetsdriftdiffusionmetric,
          title={Physics-{I}nformed {DeepONets} for drift-diffusion on metric graphs: simulation and parameter identification}, 
          author={Jan Blechschmidt and Tom-Christian Riemer and Max Winkler and Martin Stoll and Jan-F. Pietschmann},
          year={2025},
          eprint={2505.04263},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2505.04263}, 
    }


## Requirements
All codes are implemented in Python 3 and mainly rely on JAX.
The JAX version used and tested during the development is 0.4.38.

## Structure
This repository contains four Jupyter notebooks.

### 1. Generation of training data

The generation of the training data is described in `01_Generate_Data.ipynb`. It doesn't has to be executed to run the model.

### 2. Model training

Model training happens in `02_Learn_PI_DeepONet.ipynb`. It doesn't has to be executed to run the model. Weights used in the paper are contained in the directory `final_params`.

### 3. Inference and simulation

The solution of the drift-diffusion on a metric graph is implemented in `03_Simulation.ipynb`.

### 4. Solution of inverse problems

The solution of inverse problems for the drift-diffusion on a metric graph is implemented in `04_Inverse.ipynb`.

## If you want to train your own PDE?

In order to train a completely new model one has to start to implement a quantum graph example in `src/graph.py` and generate training data for inflow, inner and outflow edges, reps., using the script `01_Generate_Data.ipynb`.
Using this data, new models have to be trained using `02_Learn_PI_DeepONet.ipynb` for all edge types.
Afterwards, one can adapt `03_Simulation.ipynb` and `04_Inverse.ipynb` to solve the new quantum graph problem.

## Credits

Parts of the functions are based on the repository [https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets].
