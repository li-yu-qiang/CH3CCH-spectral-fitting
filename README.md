# CH3CCH Spectral Fitting Code

This repository provides Python scripts for fitting CH3CCH rotational spectra with a multi-Gaussian LTE model. The code was developed for the analysis presented in:

Li, Y. et al. 2026, *The Astrophysical Journal*, 998, 182,
"CH3CCH as a Thermometer in Warm Molecular Gas"
DOI: 10.3847/1538-4357/ae33bc

## Overview

The scripts fit CH3CCH spectra by modeling multiple K-ladder components simultaneously. For each source, the rotational temperature, total column density, line width, and systemic velocity are treated as global fitting parameters shared by all selected K components.

The example included in this repository fits CH3CCH J=5-4, K=0–2 emission toward G035.19−00.74.

## Files

| File                              | Description                                                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| CH3CCH_fitting.py                 | Original version of the CH3CCH fitting script.                                                                         |
| Accelerated_version.py            | Accelerated version using precomputed quantities, a partition-function lookup table, and optional numba compilation.   |
| Accelerated_version_parallel.py.  | Version prepared for parallel or batch fitting workflows.                                                              |
| c040502.cat                       | CDMS catalog file containing CH3CCH spectroscopic parameters.                                                          |
| G03519-0074_CH3CCH.dat            | Example observed spectrum used for testing and demonstration.                                                          |
| G035.pdf                          | Example fitting result for G035.19−00.74.                                                                              |

## Requirements

The code requires Python 3 and the following packages:

bash
pip install numpy scipy pandas matplotlib astropy numba


numba is optional. If it is not installed, the accelerated scripts fall back to a NumPy implementation.

## Quick start

Place the following files in the same directory:

CH3CCH_fitting.py
Accelerated_version.py
Accelerated_version_parallel.py
c040502.cat
G03519-0074_CH3CCH.dat

Run the original version:

python CH3CCH_fitting.py

Run the accelerated version:

python Accelerated_version.py

Run the parallel or batch-processing version:

python Accelerated_version_parallel.py


The example script reads G03519-0074_CH3CCH.dat, fits the CH3CCH J=5–4 K components, prints the best-fit parameters, and saves a PDF figure, for example:

G035.pdf


## Input data format

The example input spectrum G03519-0074_CH3CCH.dat should be a two-column ASCII file:

velocity intensity

where:

* velocity is in km/s;
* intensity is antenna temperature before the main-beam efficiency correction used in the script.

In the example, the intensity is divided by main beam efficiency (eta_mb = 0.225) before fitting.

## Main fitting parameters

The fitting function has the following main inputs:

python
fitting(
    J=5,
    K0=0,
    K1=2,
    velocity=velocity,
    intensity=intensity,
    v_0_init=34,
    FWHM_init=3,
    v0=(30, 40),
    sigmav=(lower_sigma, upper_sigma),
    source_name="G035.19-00.74",
    source="G035",
)


Important parameters:

| Parameter     | Meaning                                                |
| ------------- | ------------------------------------------------------ |
| J             | Upper rotational quantum number.                       |
| K0, K1        | Range of K components included in the fit.             |
| velocity      | Velocity array in km/s.                                |
| intensity     | Spectrum intensity array.                              |
| v_0_init      | Initial guess for systemic velocity in km/s.           |
| FWHM_init     | Initial guess for FWHM in km/s.                        |
| Trot          | Lower and upper bounds for rotational temperature.     |
| Ntot          | Lower and upper bounds for log10 total column density. |
| sigmav        | Lower and upper bounds for Gaussian sigma velocity.    |
| v0            | Lower and upper bounds for systemic velocity.          |
| source_name   | Source name shown in the output plot.                  |
| source        | Output filename prefix.                                |

## Output

The accelerated scripts return:

python
popt, pcov, errors


where:

text
popt[0] = T_rot
popt[1] = log10(N_tot)
popt[2] = sigma_v
popt[3] = v_lsr


The FWHM is calculated as:

python
FWHM = sigma_v * 2 * sqrt(2 * ln(2))


The example output figure shows the observed spectrum and the best-fit model.

## Citation

If you use this code, please cite:

@ARTICLE{2026ApJ...998..182L,
       author = {{Li}, Yuqiang and {Wang}, Junzhi and {Li}, Juan and {Lu}, Xing and {Zheng}, Siqi and {Ou}, Chao and {Huang}, Qian and {Santander-Garc{\'\i}a}, Miguel and {D{\'\i}az-Luis}, Jos{\'e} Jairo and {Lee}, Seokho and {Liu}, Tie and {Shen}, Zhiqiang},
        title = "{CH$_{3}$CCH as a Thermometer in Warm Molecular Gas}",
      journal = {\apj},
     keywords = {Massive stars, Star formation, Interstellar molecules, 732, 1569, 849, Astrophysics of Galaxies, Solar and Stellar Astrophysics},
         year = 2026,
        month = feb,
       volume = {998},
       number = {1},
          eid = {182},
        pages = {182},
          doi = {10.3847/1538-4357/ae33bc},
archivePrefix = {arXiv},
       eprint = {2601.19344},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026ApJ...998..182L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


## Notes

* The model assumes LTE conditions.
* The K components are fitted with Gaussian profiles.
* The selected K components share the same rotational temperature, column density, FWHM, and Vlsr.
* The CDMS catalog file must be present in the working directory.
* For fitting transitions or sources other than the included example, update the input data file, source name, initial guesses, and parameter bounds accordingly.

## Acknowledgements

The original CH3CCH fitting script was developed by Yuqiang Li. I am grateful to Yaocheng Chen for improving the computational performance of the original code and for developing the accelerated and parallel accelerated versions included in this repository.
