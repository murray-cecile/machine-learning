## HW #3 - File structure

* The writeup can be found in the top-level directory and is named HW3-Report.
* Some exploratory work demontrating the pipeline's functionality can be found in HW3-exploration. There are also two Python scripts, build_models.py and analyze_results.py, that script the data cleaning, model runs, and performance evaluation.
* The pipeline library has been split into three parts:
    1. utils.py contains the methods for reading, cleaning and preparing data
    2. exploration.py contains the methods for exploratory analysis of features
    3. pipeline.py contains the methods for building and evaluating classifiers
* The data/ directory contains the raw data.
* The output/ directory contains two supporting tables:
    1. The table contained in the report.
    2. The full table of all 1,155 performance metrics from 165 models evaluated on 7 thresholds.

