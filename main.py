# Omri Ifrah
# Models for predicting violent crime in US neighborhoods based on a variety of parameters

import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

DATA_FILE_PATH = 'communities.data'


def get_names() -> list:
    """
    find the header names in the names file and return the names in a list
    :return: list
    """
    file = open('communities.names', 'r')
    lines = file.readlines()
    names = []
    # Strips the newline character
    for line in lines:
        if "@attribute" in line:
            names.append(line.split()[1])
    return names


def prepare_data() -> pd.DataFrame:  # ex1
    """
    Preparing the data table.
    :return: pd.DataFrame
    """
    header = get_names()
    df = pd.read_csv(DATA_FILE_PATH, names=header)
    df = df.iloc[:, 5:]  # remove first 5 columns
    df = df[df.apply(lambda val: val != '?')]    # if val == '?' replace with None
    df = df.dropna(axis='columns')  # drop all columns that contains None
    df = df[df.apply(lambda val: (val <= 1) & (val >= 0))]  # if val < 0 or val > 1 replace with None
    df = df.dropna(axis='columns')  # drop all columns that contains None
    print(f"- Stayed {df.shape[0]} rows and {df.shape[1]} columns.\n")  # Printing the number of rows and columns left
    return df


def simple_linear(df: pd.DataFrame) -> pd.Series:
    """
    Finding correlations and choosing informative variables for a linear model.
    :param df: pd.DataFrame
    :return: pd.Series
    """
    dictionary = {}
    for column in df:
        if column != 'ViolentCrimesPerPop':
            x = df[column]
            y = df['ViolentCrimesPerPop']
            dictionary[column] = scipy.stats.pearsonr(x, y)[0]    # Correlation Calculation

    series = pd.Series(dictionary).sort_values(key=abs, ascending=False)
    series.abs().plot()
    plt.show()
    fig = plt.figure(figsize=(20, 10))
    for i in range(0, 12):
        x = df[series.keys()[i]]
        y = df['ViolentCrimesPerPop']
        ax = fig.add_subplot(4, 3, (i+1))
        ax.scatter(x, y)
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color='orange')
        ax.set_title(series.keys()[i])

    fig.tight_layout()
    plt.show()
    return series


def build_mlr_model(df: pd.DataFrame):
    """
    Creating a linear MLR model that uses all variables to predict the ViolentCrimesPerPop column.
    :param df: pd.DataFrame
    :return: None
    """
    X = sm.add_constant(df.drop(columns=['ViolentCrimesPerPop']))  # All columns without ViolentCrimesPerPop
    Y = df.ViolentCrimesPerPop  # That is the target column
    fit = sm.OLS(Y, X).fit()
    print(fit.summary())
    print(f"ex3 Adj. R-squared = {fit.rsquared_adj} \n")


def build_specific_mlr_model(df: pd.DataFrame, series: pd.Series):
    """
    Creating a linear model from the 12 variables with the highest correlation to ViolentCrimesPerPop.
    The Adjusted test R-squared.
    :param df: pd.DataFrame
    :param series: pd.Series
    :return: None
    """
    columns_names = series.keys()[0:12]
    X = sm.add_constant(df[columns_names])  # top 12 columns by Correlation
    Y = df.ViolentCrimesPerPop  # That is the target column
    fit = sm.OLS(Y, X).fit()
    print(fit.summary())
    print(f"ex4 Adj. R-squared = {fit.rsquared_adj}")


def main():
    data_file_name = 'communities.data'
    df = prepare_data()
    series = simple_linear(df)
    build_mlr_model(df)
    build_specific_mlr_model(df, series)


if __name__ == '__main__':
    main()

