# decision_tree_cof

# This repository contains a Python code that can predict the friction coefficient of porous AlSi10Mg and AlSi10Mg-SiC composite materials under dry-sliding conditions using a decision tree regressor.

# Dependencies

# The code requires Python 3.8 or higher to run. The code also depends on several libraries that need to be installed before running the script, such as: 
# scikit-learn, pandas, numpy, matplotlib, openpyxl, and patheffects. The installation instructions for each library can be found on their respective websites or documentation pages.


# Input/Output files

# The code reads the data from Excel files with each material's time and friction force measurements. The data files are named as follows: E_1.xlsx, E_2.xlsx, E_3.xlsx, SE_1.xlsx, SE_2.xlsx, and SE_3.xlsx, where E stands for AlSi10Mg and SE stands for AlSi10Mg-SiC composite. Each file has two columns: TIME and FF, which represent the time in seconds and the friction force in Newtons, respectively.

# The code saves the results and outputs in different formats, such as:

# Average_COF_E.xlsx or Average_COF_SE.xlsx: an Excel file that contains the average COF for each material.
# Average_time_E.xlsx or Average_time_SE.xlsx: an Excel file that contains the average time for each material.
# COF_E_SE.xlsx: an Excel file that contains the concatenated data for both materials.
# E_performance_metrics.txt or SE_performance_metrics.txt: a text file containing both sets' performance metrics.
# pred_COF_E.png or pred_COF_SE.png: a png file that shows the scatter plot of the actual vs predicted COF as a function of time for both sets.
# test_val_data_E.xlsx or test_val_data_SE.xlsx: an Excel file that contains the test and validation sets for each material.

# Installation and usage

# To run the code:

# Have the following files in the same directory as the code: E_1.xlsx, E_2.xlsx, E_3.xlsx, SE_1.xlsx, SE_2.xlsx, and SE_3.xlsx. These files contain the time and friction force measurements for each material and dataset.
# Open a command prompt or a terminal window and navigate to the root folder.
# Type: python decision_tree_cof.py

# License
# This project is licensed under the MIT License (see the LICENSE file for details).









