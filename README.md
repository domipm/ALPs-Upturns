# ALPs-Upturns (alpsup)

**alpsup** is a Python package and analysis pipeline developed for performing joint analyses of Fermi-LAT and H.E.S.S. gamma-ray observations to search for spectral signatures of photon–axion-like particle (ALP) conversions.

The pipeline is built primarily on the open-source [Gammapy](https://github.com/gammapy/gammapy) and [FermiPy](https://github.com/fermiPy/fermipy) libraries, with additional use of [GammaALPs](https://github.com/me-manu/gammaALPs) for photon-ALP propagation simulations and [EBLTable](https://github.com/me-manu/ebltable) for extragalactic background light attenuation models.

This code accompanies my [master's thesis](thesis/main.pdf) "Searches for Spectral Upturns due to Photon–Axion-Like Particle Conversions using H.E.S.S. and Fermi-LAT Observations.", which includes a full overview of the physics background, methodology, and obtained results. 

This repository contains:
- the complete analysis pipeline developed during my MSc thesis,
- reusable Python modules (alpsup),
- analysis and simulation scripts,
- configuration files,
- the written thesis describing the methodology and results.

NOTE: This repository is currently a work in progress. The original research code used for the thesis is being refactored into a self-contained, reproducible analysis pipeline with improved documentation, modularity, and installation procedures.

## Overview and theoretical background

<img width="1539" height="572" alt="Captura de pantalla 2026-07-15 a las 16 17 34" src="https://github.com/user-attachments/assets/36651915-d858-469b-894c-b2561176451c" />

## Repository structure

The repository is structured in the following way.

```text
.
├── configs/             # Configuration flies and their templates
│   └── fermi_config.yaml      # Configuration template for FermiPy analysis
├── data/                # All required data for analysis
    ├── fermipy-data/          # Data used by FermiPy (4FGL catalog, diffuse background templates, not included))
    ├── gammapy-data/          # Data available with GammaPy 2.0 (sample data, not, included)
    ├── flat-data/             # Data from Fermi-LAT used for the analysis of all sources (not included)
    │   ├── TARGET/                  # Per-target subfolder
    │   ├── get_files.sh             # Automation shell script for downloading public Fermi-LAT data
    │   └── spacecraft.fits          # Global spacecraft data file used for analysis (encompassing full observational time selection across all sources)
    ├── hess-data/             # Data from H.E.S.S. used for the analysis of all sources (not included)
    │   ├── TARGET/                  # Per-target subfolder
    │   └── get_runlist.sh           # Bash script to generate / check runlist versus 
    └── README_DATA.md         # README file explaining the data folder structure
├── envs/                # Conda environment definitions
├── results/             # Generated results for all sources
    ├── ALPs/                    # ALP simulation / interpolation results (global)
    ├── <TARGET>/                # Per-target subfolder
    ├── baseline/                      # Per-block subfolders
    ├── block1/                        
        ├── alps/                            # ALP simulation / interpolation results (per-source)
        ├── ebl/                             # Per-EBL subfolders
            ├── dominguez/
                ├── gamma-out/
                ├── logs/
                ├── plots/
                └── ...
            └── ...
        ├── fermi-out/                      # FermiPy output
        ├── gamma-out/                      # GammaPy output
        ├── logs/                           # Log files
        ├── plots/                          # Final plots
        └── fermi_config.py                 # FermiPy configuration file
    └── ...
├── scripts/             # Analysis and simulation scripts
    ├── bash/                  # Automation shell scripts
    ├── cluster/               # Data reduction wrappers and automation scripts for computational cluster   
    ├── ...                    # Individual Python scripts for performing each step of the analysis
├── sources/             # Source catalogues and metadata
├── src/alpsup/          # Core Python package
├── thesis/              # MSc thesis and LaTeX code
├── README.md
├── .gitignore
└── pyproject.toml
```

## Data reduction and analysis pipeline

The analysis is divided into the following steps.

### Data reduction and access / availability

Depending on the instrument and availability of the data, the data can be used directly with a minimal reduction process, or a more extensive pipeline using internal tools. 

- Fermi-LAT: data is publicly available via the LAT Data Server: https://fermi.gsfc.nasa.gov/ssc/data/access/. In this case, the data can be directly imported into FermiPy, which will then perform a series of setup steps to generate the required `*.fits` files for use in GammaPy.
- H.E.S.S.: data is not publicly available, except for data releases (available via https://hess-experiment.eu/releases/). In this case, the data must be first prepared using internal tools developed by the collaboration to generate the `*.fits` files compatible with GammaPy. Some of the auxiliary scripts developed for this are available in the `scripts/cluster` directory. The proprietary H.E.S.S. software they interface with is not part of this repository.

The final selection of sources, and their relevant information, can be found in the `sources` directory, with auxiliary scripts to produce visualizations of the data.

### Data analysis, simulations, and likelihood analysis

The data analysis pipeline is run sequentially with the available scripts in the `scripts` folder.

1. Lightcurve generation for H.E.S.S. data and time variability analysis -- `scripts/hess_lightcurve.py` produces H.E.S.S. light curves and identifies emission states using Bayesian Blocks and spectral evolution analyses.
2. Fermi-LAT analysis with FermiPy and GammaPy -- `scripts/fermi_analysis.py` and `scripts/flat_analysis.py` runs the FermiPy and subsequent GammaPy analysis on Fermi-LAT data, based on the generated configuration file.
3. H.E.S.S. analysis with GammaPy -- `scripts/hess_analysis.py` runs the GammaPy analysis of H.E.S.S. data, including EBL attenuation and a treatment of systematic uncertainties with additional bias parameters.
4. Joint Fermi-LAT and H.E.S.S. analysis with GammaPy -- `scripts/joint_analysis.py` runs the GammaPy analysis of the combined Fermi-LAT and H.E.S.S. dataset.
5. Upturn modeling -- `scripts/model_upturns.py` implements an upturn spectral model and a grid-search approach to compute the likelihood of an upturn being present in the data, by performing spectral fits of this upturn spectral model across a physically motivated parameter space.
6. Simulation of photon-ALP conversions -- `scripts/model_alps.py` performs numerical simulations of the gamma-ray photon propagation across the intergalactic medium, including EBL absorption and conversions into ALPs, to compute the expected spectral upturns across a range of ALP couplings to the electromagnetic field.
7. Interpolation and likelihood analysis -- `scripts/interp_alpupturn.py` interpolates simulated ALP-induced upturns with spectral upturn modeling grids to compute the likelihood of photon-ALP conversions inducing spectral upturns in the observed data.

Auxiliary bash scripts in `bash` are used to automatically run the scripts for a single (or all) sources, such as `run_scripts.sh`.

## Installation and usage

Two separate conda environments are required to run the full analysis process (with the environment dependencies listed in the `envs` directory and relevant `env_*.yaml` files). The auxiliary package `alpsup` must be installed as a local, editable package in both environments to access scripts required for the analysis.
```bash
conda env create -f envs/env_alps-upturns.yaml
conda activate alps-upturns
pip install -e .
```

The main environment used for the GammaPy analysis, as well as the GammaALPs simulations, is given by `env_alps-upturns.yaml`, while the environment required to run the FermiPy analysis is given by `env_alps-upturns-fermipy.yaml`.

In order to run a script, it is sufficient to then execute it with Python, providing the required arguments (depending on the script). For example:
```bash
python scripts/hess_analysis.py --source PKS2155-304 --bblock baseline --ebl dominguez
```
