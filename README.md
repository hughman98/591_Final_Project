# 591 Final Project

J. Hugh Wright

Vignesh Muthukumar

Gabriel Silva de Oliviera

This project used the following datasets collected from DataShop:

1. https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=92
2. https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=120
3. https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=339

Sadly, these datasets are too large to be included within this repository. However, we do include CSV files containing the data we extracted from them.

The following is a list of the files included in this rep, and what they were used for.

1. baseline_model.py: A python script for training and testing our non-neural network predictive models.
2. correlationcalculation.py: A pythons script for calculating the Pearson's correlation coefficients and p-values of our data.
3. dataHotEncoding.csv: A CSV file containing the final form of our data, with all KCs one hot encoded. Used to train the predictive models.
4. Graphs.xlsx: An excel file that we used to create our data visualizations.
5. hotencoding.py: A python script for one hot encoding our knowledge components. Produced the "dataHotencoding.csv" file.
6. language_features.py: A python script for extracting language features and adding them to our data set. Produced the "Language_Processed.csv" file in the ProcessedData folder
7. lowfrequencycalculation.py: A python script for detecting low frequency words. Due to time constraints, we ultimately did not use this data in our analysis.
8. neural_networks.py: A Python script for training and testing our neural network models.
9. preprocess.py: A simple python script to extract information about individual questions from our three original datasets. Produced the "More_Processed_Data.csv" in the ProcessedData folder.




