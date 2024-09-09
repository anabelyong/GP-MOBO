# Multi-Objective Bayesian Optimization with Independent Tanimoto Kernel Gaussian Processes for Diverse Pareto Front Exploration (README.md in construction)

GP-MOBO is a novel multi-objective Bayesian Optimization (MOBO) algorithm designed to optimize molecular properties using Gaussian Processes (GPs). Leveraging independent Tanimoto kernel GPs for each molecular objective, the model effectively explores the Pareto frontier, balancing exploration and exploitation to identify high-quality, diverse candidate molecules.

**Key Features:**
- **Independent Tanimoto Kernel GPs:** Models each molecular objective separately, capturing the full dimensionality of molecular fingerprints without reducing complexity.
- **Efficient Pareto Front Exploration:** Utilizes the Expected Hypervolume Improvement (EHVI) acquisition function, ensuring superior coverage of the chemical search space. 
- **Scalable & Computationally Efficient:** The model scales well for large datasets and is optimized for multi-objective tasks, making it suitable for drug discovery and molecular design. 

Python Scripts to Run: 
1) Dockstring Toy MPO Setup
2) GUACAMOL MPO Setup
For DockSTRING Toy MPO Setup, go to ```dockstring-test-implementation``` branch, run for 3 experiments:
```
python ehvi_mc_3_trials.py
```
or
```
python ehvi_mc.py
```
For GUACAMOL MPO Setup, go to ```guacamol-test```branch implementation, run:
```
python ehvi_{mpo_name}.py
```
Example: 
```
python ehvi_fexofenadine.py
```

Datasets to Download: 
1) DOCKSTRING (https://github.com/dockstring/dockstring)
2) GUACAMOL: EXTRACTED FROM GUACAMOL BENCHMARK (https://github.com/BenevolentAI/guacamol)
```
pip install dockstring
```

## Pacakge Versions: 

**Running the code requires:**

- ```KERN_GP``` which consists of a minimal kernel-only GP package from https://github.com/AustinT/kernel-only-GP
- ```Numpy```
- ```Rdkit```
- ```PyTDC``` which consists the oracle functions for all objectives required. https://github.com/mims-harvard/TDC

**Running for code comparison to existing methods requires:**
- https://github.com/AustinT/basic-mol-bo-workshop2024


## Development

Please use pre-commit for code formatting / linting.
