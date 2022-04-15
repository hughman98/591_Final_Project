import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


def read_from_file(file_name):
    processed_data = pd.read_csv(file_name)
    return processed_data


def split_x_y(input_data):
    x = input_data.drop(['average time to solve', 'Question', 'person_name_boolean', 'Unnamed: 17', 'num students',
                         'tot_time_spent', 'num_success', 'num_failed', 'success_rate'], axis=1)
    y = input_data['average time to solve']
    return x, y


def get_coefficients(x_train_input, model):
    # match column names to coefficients
    for coef, col in enumerate(x_train_input.columns):
        print(f'{col}:  {model.coef_[coef]}')


# function to get cross validation scores
def get_cv_scores(model, x_train_input, y_train_input, scoring):
    scores = cross_val_score(model, x_train_input, y_train_input, cv=5, scoring=scoring)

    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')


def linear_regression_model(x_train_input, y_train_input, scoring):
    print("Trying linear regression")
    linear_regression = LinearRegression()
    linear_regression.fit(x_train_input, y_train_input)
    # get cross val scores
    get_cv_scores(linear_regression, x_train_input, y_train_input, scoring)


def ridge_regression_model(x_train_input, y_train_input, is_ridge_grid, scoring):
    ridge = Ridge(alpha=1).fit(x_train_input, y_train)

    if is_ridge_grid:
        print("Trying ridge regression with grid search cv on ALPHA")
        alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        param_grid = dict(alpha=alpha)
        grid = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring=scoring, verbose=1,
                            n_jobs=16)
        grid_result = grid.fit(x_train_input, y_train_input)
        print('Best Score: ', grid_result.best_score_)
        print('Best Params: ', grid_result.best_params_)
        print("\n")
        get_coefficients(x_train_input, ridge)
    else:
        print("Trying ridge regression")
        get_cv_scores(ridge, x_train_input, y_train, scoring)


def lasso_regression_model(x_train_input, y_train_input, is_lasso_grid, scoring):
    lasso = Lasso(alpha=1).fit(x_train_input, y_train_input)

    if is_lasso_grid:
        print("Trying lasso regression with grid search cv on ALPHA")
        alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        param_grid = dict(alpha=alpha)
        grid = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring=scoring, verbose=1,
                            n_jobs=16)
        grid_result = grid.fit(x_train_input, y_train_input)
        print('Best Score: ', grid_result.best_score_)
        print('Best Params: ', grid_result.best_params_)
        print("\n")
        get_coefficients(x_train_input, lasso)
    else:
        print("Trying lasso regression")
        get_cv_scores(lasso, x_train_input, y_train_input, scoring)


def elastic_net_model(x_train_input, y_train_input, is_elastic_grid, scoring):
    elastic_net = ElasticNet(alpha=1, l1_ratio=0.5).fit(x_train_input, y_train_input)
    if is_elastic_grid:
        print("Trying ElasticNet with grid search cv on ALPHA, L1_RATIO")
        alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        param_grid = dict(alpha=alpha, l1_ratio=l1_ratio)
        grid = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring=scoring, verbose=1,
                            n_jobs=16)
        grid_result = grid.fit(x_train_input, y_train_input)
        print('Best Score: ', grid_result.best_score_)
        print('Best Params: ', grid_result.best_params_)
        print("\n")
        get_coefficients(x_train_input, elastic_net)
    else:
        print("Trying ElasticNet")
        get_cv_scores(elastic_net, x_train_input, y_train, scoring)


if __name__ == '__main__':
    data = read_from_file("dataHotEncoding.csv")
    # print(data.shape)
    # print(data.columns)

    # nan_list = data.isnull().sum()
    # for index, value in nan_list.items():
    #     print(index, value)

    # print("NaNs in dataframe ", data.isnull().sum())
    # break the data into X and y
    x, y = split_x_y(data)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # print(x_train.shape)
    # print(x_test.shape)

    # Try linear regression normally and look at the cv scores to make an evaluation about model performance
    linear_regression_model(x_train, y_train, 'r2')
    print("***************************************************************************************")

    # Try Ridge regression normally and look at the cv scores to make an evaluation about model performance
    ridge_regression_model(x_train, y_train, False, 'r2')
    print("***************************************************************************************")

    # Now to try and improve the R2 score by performing grid search on ridge regression
    ridge_regression_model(x_train, y_train, True, 'r2')
    print("***************************************************************************************")

    # Try Lasso regression and evaluate performance
    lasso_regression_model(x_train, y_train, False, 'r2')
    print("***************************************************************************************")

    # Try improving R2 score by performing grid search on lasso regression
    lasso_regression_model(x_train, y_train, True, 'r2')
    print("***************************************************************************************")

    # Train elastic net regression model with default alpha=1 and l1_ratio=0.5
    elastic_net_model(x_train, y_train, False, 'r2')
    print("***************************************************************************************")

    # Try improving score by performing grid search on alpha and l1 in elastic net
    elastic_net_model(x_train, y_train, True, 'r2')
    print("***************************************************************************************")
