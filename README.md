# Gust Response Predictions of a Very Flexible Wing Model

This repository contains the scripts and code used for the SHARPy simulations obtained for the results shown in the journal paper

[1] Duessler, S., Mertens, C., & Palacios, R. Gust Response Predictions of a Very Flexible Wing Model. AIAA Journal, 2025. (https://doi.org/10.2514/1.C038332)

In this paper, nonlinear aeroelastic simulations are benchmarked against wind tunnel experiments of a very flexible wing [2]. To do so, sectional force corrections are employed in static and dynamic nonlinear aeroelastic simulations to capture low-Reynolds-number effects and the static lift deficiency due to the onset of flow separation. 

<video src="assets/gust_vanes.mp4" controls width="100%"></video>
<p align="center">
<strong>Visualization of the induced deformation of a very flexible wing by the gust vane wakes.</strong>
</p>


As an alternative to a frozen gust model, a simulation of the unsteady inflow to the Delft-Pazy wing that is produced by the gust vanes in the wind tunnel is explored. Results show a considerable influence of the wing's presence on the upstream gust velocity. The structural response, however, differs only slightly between the two gust models, confirming that the uniform gust assumption remains valid for moderately large deflections (up to 24% of the wingspan).

These enhancements have been implemented to the nonlinear aeroelastic simulation environment [SHARPy](http://github.com/imperialcollegelondon/sharpy) which is included in this repository as a submodule, as well as the SHARPy version of the [Pazy model](http://github.com/ngoiz/pazy-model). The scripts in this repo include the pre- and postprocessors for the results in the AIAAJ paper.

## Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules <repository-url>
cd pazy-gust-response
```

If you already cloned without `--recurse-submodules`, initialise the submodules with:

```bash
git submodule update --init --recursive
```

### 2. Install SHARPy

Follow the [SHARPy installation guide](https://ic-sharpy.readthedocs.io/en/latest/content/installation.html) to create the `sharpy` conda environment. The SHARPy version extensively last tested with this code is version 2.4.

### 3. Activate the environment and install packages

```bash
conda activate sharpy
pip install -e lib/sharpy
pip install -e lib/pazy-model
```

## Structure of this Repository

* `lib/`: SHARPy and Pazy model submodules.
* `simulations/`: Scripts to generate and run SHARPy cases, including steady wing deformation, gust response, and convergence studies.
* `postproc/`: Post-processing scripts for extracting and plotting results.
* `experimental_data/`: Experimental result from [2] for comparison.

## Running Simulations

Activate the SHARPy environment, navigate to `simulations/`, and run the desired script, e.g.:

```bash
conda activate sharpy
cd simulations
python run_gust_response.py
```

## References
[1] Duessler, S., Mertens, C., & Palacios, R. Gust Response Predictions of a Very Flexible Wing Model. AIAA Journal, 2025. (https://doi.org/10.2514/1.C038332)

[2] Mertens, C., Costa Fernández, J., Sodja, J., Sciacchitano, A., and van Oudheusden, B. Nonintrusive Experimental Aeroelastic Analysis of a Highly Flexible Wing. AIAA Journal, 2023. (https://doi.org/10.2514/1.J062476)

## Copyleft

We are happy to share our efforts with the community and we welcome contributions to the code base. If you found this dataset useful we would kindly ask you to cite the paper [1] in any publications or reports based on it.


