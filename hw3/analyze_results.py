#==============================================================================#
# HW #3 - ANALYZE DONORSCHOOSE RESULTS
# CAPP 30254
#
# Cecile Murray
#==============================================================================#


import pandas as pd 

if __name__ == "__main__":
    
    # read in model performance results
    results = pd.read_csv('output/run_all.csv')

    # segment to just the 165 where the threhold was 5%
    results5 = results.loc[results['Threshold'] == 0.05]

    # Create table showing precision, recall, and accuracy 
    metrics = ['classifier', 'params', 'Train/Test Split ID', 'Precision', 'Recall', 'Accuracy', 'AUC_ROC Score']
    results5[metrics].groupby(['classifier', 'params', 'Train/Test Split ID']).mean().to_csv('output/Memo_Table.csv')