## README - HW #3

The goal of this homework assignment is to fix and update your ML pipeline.

The problem is to predict if a project on donorschoose will not get fully funded within 60 days of posting. This prediction is being done at the time of posting so you can only use data available to you at that time. The data is a file that has one row for each project posted with a column for date_posted (the date the project was posted) and a column for date_fully_funded (the date the project was fully funded - assumption for this assignment is that all projects were fully funded eventually). The problem and data are adopted from here: https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data.

The goal of this assignment is to improve the pipeline by adding more classifiers, experimenting with different parameters for these classifiers, adding additional evaluation metrics, and creating a temporal validation function to create training and test sets over time.

### File structure

* The writeup can be found in the top-level directory and is named HW3-Report.
* The script build_models.py scripts the data cleaning, model runs, and performance evaluation.
* The pipeline/ directory contains the three parts of my pipeline:
    1. utils.py contains the methods for reading, cleaning and preparing data
    2. exploration.py contains the methods for exploratory analysis of features
    3. modeling.py contains the methods for building and evaluating classifiers
* The data/ directory contains the raw data.
* The output/ directory contains two supporting tables:
    1. The table contained in the report.
    2. The full table of all 1,155 performance metrics from 165 models evaluated on 7 thresholds.

### How to run the pipeline

The two key pieces to running the pipeline are the build_models.py script and the config.yml file. The script build_models.py uses the parameters set in the config file to read and process data and build and evaluate models.
