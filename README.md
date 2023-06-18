# Grid Codes versus Multi-Scale, Multi-Field Codes for Space

## 1. General Information

This repository contains the code used for the investigations of multi-scale, multi-field (MSMF) place codes and their comparison against grid codes which is presented in Dietrich et al. [[1]](#4-references). This MSMF code was recently found by Eliav et al. [[2]](#4-references) during experiments with bats flying in a 200 m tunnel. In their experiments, they found that the place cells of the bats not just maintained a single firing field in the environment but multiple ones of highly varying sizes (0.5-15 m). In a theoretical analysis, the authors showed that the newly found MSMF code is superior to a "traditional" single-scale, single-field code (SSSF). 

Within our analysis [[1]](#4-references), we demonstrated, that it is indeed superior to a SSSF code with respect to the accuracy of the positional decoding but is inferior to the grid code found in the entorhinal cortex of rats [[3]](#4-references). While both codes are able to achieve a positional decoding error of 0.0, encoding the position of an agent in a 200 m long environment with 400 bins (bin size 0.5 m), the grid code requires fewer neurons and energy to achieve the same result. This is contingent upon the optimal distribution of the fields in a grid code [[4]](#4-references), which on the other hand also leads to a less robust system than the MSMF networks, which can even cope with up to 25 % drop-out while still maintaining a positional decoding error of less than one meter. 
Further analyses of the network dynamics also revealed that the proposed topology of MSMF cells by Eliav et al. [[2]](#4-references) does not, in fact, result in a continuous attractor network. More precisely, the simulated networks do not maintain activity bumps without position specific input. The multi-scale, multi-field code, therefore, seems to be a compromise between a place code and a grid code that invokes a trade-off between accurate positional encoding and robustness.

This repository provides all scripts for running the simulations, optimizations and evaluations of MSMF, SSSF and grid models which were performed for the publication [[1]](#4-references). Beyond that, all results of the optimizations and evaluations are included in the `data` folder in form of .csv files.


## 2. Installation

The entire project is implemented in Python. All necessary packages needed to run the simulations, optimizations or evaluations are included in the `requirements.txt` file within the root of the package. The simulations run on a standard CPU and do not require a GPU.  


## 3. Usage

The package contains models for the fixed and dynamic MSMF models, as well as the grid code and a standard 1D line attractor (single-scale, single-field (SSSF)). All models as well as the code for simulating them are included in the `msmfcode/models/cann.py` file. Below you can find a table with the most important files using these models for performing simulations, evaluations or optimizations.

| File                        | Description                                                                                                             |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `eval_all_msmf.ipynb`       | Run simulations of a network with one configuration of parameters.                                                      |
| `eval_multi-run_msmf.ipynb` | Run evaluations of a network with multiple configurations, e.g. changing the drop-out rate within a range between runs. |
| `cga_msmf.ipynb`            | Perform optimizations using any of the provided models.                                                                 |

All configuration files necessary for evaluations or optimizations are located in the `config` folder.

Detailed and commented examples of how to run the simulations and optimizations will be provided in the near future.


## 4. References

[1] &emsp; R. Dietrich et al., "[_Grid Codes versus Multi-Scale, Multi-Field Place Codes for Space_]()" bioRxiv, June 2023.

[2] &emsp; T. Eliav et al., “[_Multiscale representation of very large environments in the hippocampus of flying bats_](https://science.sciencemag.org/content/372/6545/eabg4020)” Science, May 2021.

[3] &emsp; T. Hafting et al., “[_Microstructure of a spatial map in the entorhinal cortex_](https://www.nature.com/articles/nature03721)” Nature, Aug. 2005.

[4] &emsp; A. Mathis et al., “[_Probable nature of higher-dimensional symmetries underlying mammalian grid-cell activity patterns_](https://doi.org/10.7554/eLife.05979)” eLife, Apr. 2015.
