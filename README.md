# To Reproduce All the Results

We use python 3.10 and pytorch 1.13 for development.


### How to link Python to R
We use the python package `rpy2` to call the [`dHSIC` package from R](https://CRAN.R-project.org/package=dHSIC).

The R version must be 4.2.0 or higher.

On Linux check what the R path is: `/usr/bin/R` or `/usr/local/bin/R`

If python code `utils.install_packages("dHSIC")` cannot install the R package `dHSIC`, install it manually using R console.


### To Reproduce Results in Section 6.1

`cd CAREFL_H`

`bash simulated_tests_synthetic_datasets.sh`


### To Reproduce Results in Section 6.2

`cd CAREFL_H`

`bash benchmark_tests_SIM.sh`


### To Reproduce Results in Section 6.3

`cd CAREFL_H`

`bash benchmark_tests_Tubingen_CEpairs.sh`


### To Reproduce Results in Table 1

`cd CAREFL_H`

`bash simulated_tests_teaser_table.sh`


### To Reproduce Results in the Suitability Table

`cd CAREFL_H`

`bash suitability.sh`


### Plot the Figures in Section 6
After getting the result CSV files by running the scripts above, one can call `util_functions.py` to draw the figures. 

(Please make sure you have all the result CSV files, otherwise it may throw a File Not Found Error. If you don't have all the result CSV files, please comment out corresponding part of the code in `util_functions.py` to ignore the corresponding figures.)
