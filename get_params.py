# This is a sample Python script.

import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)


def find_parameters(X, y):
    regressor = MLPRegressor(max_iter=10000, early_stopping=True)
    parameters = {
        'hidden_layer_sizes': [(30, 30, 30), (30, 30, 30, 30), (30, 30, 30, 10), (50, 50, 50, 50), (100, 50, 50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01, .05, .1],
        'learning_rate': ['adaptive'],
    }

    search = GridSearchCV(regressor, parameters, n_jobs=-1, cv=5, scoring="r2")
    search.fit(X, y)

    features = ['tot_time_spent', 'num_success']

    # This was just a test for myself to see how plotting PartialDependence graphs works. Turns out: super easy
    # graph = PartialDependenceDisplay.from_estimator(search, X, features)

    # plt.show()

    return search.best_params_, search.best_score_


# Preprocesses data and returns the "input" data and "ouput" data that is used to train the regressors.
# Does not fetch burgers.
#
# The code for this function is based on Assignment 3 that I completed on date.
#
def get_in_out():
    print('importing csv...')
    input = pd.read_csv("dataHotEncoding.csv")

    # set target to be average time
    output_t = input['average time to solve']
    output_s = input['success_rate']

    # Remove "output" columns and other non-desirable columns from training set
    input = input.drop(['average time to solve'], axis=1)
    input = input.drop(['success_rate'], axis=1)

    input = input.drop(['Question'], axis=1)
    input = input.drop(['tot_time_spent'], axis=1)
    input = input.drop(['num students'], axis=1)
    input = input.drop(['num_success'], axis=1)
    input = input.drop(['num_failed'], axis=1)

    t_attributes = ['person_name_count',  # Columns to consider for average time
                    # 'person_name_boolean',
                    'conjunction_phrase_count',
                    'preposition_phrase_count',
                    'math_symbols_count',
                    'Sum-Of-Interior-Angles-more-than-3-Sides',
                    'Mean',
                    'Unit-Conversion',
                    'Inducing-Functions',
                    'person_name_count',
                    'conjunction_phrase_count',
                    'Percent-Of',
                    'Rate',
                    'Pattern-Finding',
                    'Venn-Diagram',
                    'Discount',
                    'Finding-Percents',
                    'Divide-Decimals',
                    'Square-Root',
                    'Subtraction',
                    'Fraction-Division',
                    'Interpreting-Numberline',
                    'Probability',
                    'Sum-of-Interior-Angles-Triangle',
                    'Equivalent-Fractions-Decimals-Percents',
                    'Addition',
                    'num_success',
                    'Fraction-Decimals-Percents',
                    'Integers']

    s_attributes = ['person_name_count',  # Columns to consider for success rate
                    # 'person_name_boolean',
                    'conjunction_phrase_count',
                    'preposition_phrase_count',
                    'math_symbols_count',
                    'Integers',
                    'Fraction-Decimals-Percents',
                    'Meaning-of-PI',
                    'Sum-of-Interior-Angles-Triangle',
                    'Equivalent-Fractions-Decimals-Percents',
                    'Statistics',
                    'Fractions',
                    'Circle-Graph',
                    'Congruence',
                    'Isosceles-Triangle',
                    'Least-Common-Multiple',
                    'Reciprocal',
                    'Properties-of-Geometric-Figures',
                    'Fraction-Division',
                    'Divide-Decimals',
                    'Symbolization-Articulation',
                    'Linear-Area-Volume-Conversion',
                    'Rate',
                    'Of-Means-Multiply',
                    'Combinatorics',
                    'Discount',
                    'Evaluating-Functions',
                    'Increasing_Percent_(Sales_Tax)',
                    'Percent-Of',
                    'Sum-Of-Interior-Angles-more-than-3-Sides',
                    'Inducing-Functions',
                    'Area-of-Circle',
                    'Venn-Diagram',
                    'Perimeter',
                    'Unit-Conversion',
                    'Area']

    input_t = input.loc[:, input.columns.intersection(t_attributes)]
    input_s = input.loc[:, input.columns.intersection(s_attributes)]

    print('Imported!')
    return input_t, output_t, input_s, output_s


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x1, y1, x2, y2 = get_in_out()

    params, score = find_parameters(x2, y2)
    print('Best Params:', params)

    print('Score:', score)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
