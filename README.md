# To Reproduce All the Results

The development was done with:
- Ubuntu 20.04.5 LTS
- python 3.10.4
- pytorch 1.12
- R version 4.2.2 Patched (2022-11-10 r83330)
- dHSIC 2.1


### How to link Python to R
We use the python package `rpy2` to call the [`dHSIC` package from R](https://CRAN.R-project.org/package=dHSIC).

The R version must be 4.2.0 or higher.

On Linux check what the R path is: `/usr/bin/R` or `/usr/local/bin/R`

If python code `utils.install_packages("dHSIC")` cannot install the R package `dHSIC`, install it manually using R console.


### To Reproduce Results

`cd CAREFL_H`

`bash simulated_tests_teaser_table.sh`

`bash simulated_tests_synthetic_datasets.sh`

`bash benchmark_tests_SIM.sh`

`bash benchmark_tests_Tubingen_CEpairs.sh`

`bash suitability.sh`


### Plot the Figures in Experiments Section
After getting the result CSV files by running the scripts above, one can call `util_functions.py` to draw the figures. 

(Please make sure you have all the result CSV files, otherwise it may throw a File Not Found Error. If you don't have all the result CSV files, please comment out corresponding part of the code in `util_functions.py` to ignore the corresponding figures.)
