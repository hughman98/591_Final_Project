# This is a sample Python script.

import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def find_parameters(X, y):
    regressor = MLPRegressor(max_iter=20)
    parameters = {
        'hidden_layer_sizes': [(30, 30, 30, 30), (30, 30, 30, 10), (20, 20, 20, 20), (30, 30, 30), (20, 20, 20),
                               (30, 20, 20), (90, 10)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01, .05],
        'learning_rate': ['constant', 'adaptive'],
    }

    search = GridSearchCV(regressor, parameters, n_jobs=-1, cv=5)
    search.fit(X, y)

    features = ['tot_time_spent', 'num_success']

    # This was just a test for myself to see how plotting PartialDependence graphs works. Turns out: super easy
    #graph = PartialDependenceDisplay.from_estimator(search, X, features)

    #plt.show()

    return search.best_params_


# Preprocesses data and returns the "input" data and "ouput" data that is used to train the regressors.
# Does not fetch burgers.
#
# The code for this function is based on Assignment 3 that I completed on date.
#
def get_in_out():
    print('importing csv...')
    input = pd.read_csv("More_Processed_Data.csv")

    # set target to be average time
    output = input['average time to solve']

    # Remove average time and non-desirable columns from training set
    input = input.drop(['average time to solve'], axis=1)
    input = input.drop(['Question'], axis=1)

    input = input.drop(['KC3'], axis=1)  # Always empty

    KC2 = input.KC2.to_list()
    input = input.drop(['KC2'], axis=1)  # Always empty

    for i in set(KC2):
        new_col = []
        for j in KC2:
            if j == i:
                new_col.append(1)
            else:
                new_col.append(0)
        if i not in input:
            input[i] = new_col
        else:
            print('Something has gone very wrong here.')
            exit(1)

    KC1 = input.KC1.to_list()
    input = input.drop(['KC1'], axis=1)

    for i in set(KC1):
        new_col = []
        for j in KC1:
            if j == i:
                new_col.append(1)
            else:
                new_col.append(0)
        if i not in input:
            input[i] = new_col
        else:
            for j in range(len(input[i])):
                if input[i][j] == 1 and new_col[j] == 0:
                    new_col[j] = 1
            input[i] = new_col

    KC0 = input.KC0.to_list()
    input = input.drop(['KC0'], axis=1)

    for i in set(KC0):
        new_col = []
        for j in KC0:
            if j == i:
                new_col.append(1)
            else:
                new_col.append(0)
        if i not in input:
            input[i] = new_col
        else:
            for j in range(len(input[i])):
                if input[i][j] == 1 and new_col[j] == 0:
                    new_col[j] = 1
            input[i] = new_col

    print('Imported!')
    input = input.loc[:, input.columns.notna()]
    return input, output


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x, y = get_in_out()
    print(x.columns)
    print(y)
    print('Best Params:', find_parameters(x, y))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
