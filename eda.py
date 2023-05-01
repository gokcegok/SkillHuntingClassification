import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def check_df(dataframe, n_head=10, n_tail=10):
    """
    This function prints basic information about given dataframe.
        shape, data types, head of dataframe, tail of dataframe,
        null value counts

    Parameters
    ----------
    dataframe: pandas.DataFrame

    n_head: integer
            number of rows for printing head of df

    n_tail: integer
            number of rows for printing tail of df

    Returns
    -------
    """

    print("\n------------- Shape -------------\n")
    print(dataframe.shape)
    print("\n------------- Data Types -------------\n")
    print(dataframe.dtypes)
    print("\n------------- Head -------------\n")
    print(dataframe.head(n_head))
    print("\n------------- Tail -------------\n")
    print(dataframe.tail(n_tail))
    print("\n------------- NA -------------\n")
    print(dataframe.isnull().sum())
    print("\n------------- Percentages -------------\n")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    """
    This function prints value counts and class ratio of given
    categorical variable in the given dataframe.

    Parameters
    ----------
    dataframe: pandas.DataFrame

    col_name: string
              the name of the categorical variable
    plot: boolean
          If plot == True:
            plot bar plot of the categorical variable

    Returns
    -------
    """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}), "\n")

    if plot:

        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, num_col, plot=False):
    """
    This function prints descriptive statistics
    of the given numeric variable in the given dataset.

    Parameters
    ----------
    dataframe: pandas.DataFrame

    num_col: string
             the name of the numeric column
    plot: boolean
          If plot == True:
            plot bar plot of the categorical variable

    Returns
    -------
    """

    # quantiles = np.arange(0.05, 1.05, 0.05)
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)

    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.ylabel("freq")
        plt.title("hist of " + num_col)
        plt.show(block=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    This function returns names of categorical, numeric and categorical looking
    cardinal variables in the given dataset. And prints a brief report about dataset.

    Parameters
    ----------
    dataframe: pandas.DataFrame

    cat_th: int, float
            threshold value for numeric looking categorical variables

    car_th: int, float
            threshold value for categorical looking cardinal variables

    Return
    ------
    cat_cols: list
        list of categorical variables
    num_cols: list
        list of numeric variables
    cat_but_car: list
        list of categorical looking cardinal variables

    Notes
    -----
    len(cat_cols) + len(num_cols) + len(cat_but_car) = len(df)
    num_but_cols(numeric looking categorical variables) is in cat_cols
    """

    cat_cols = [col for col in dataframe.columns if
                str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print("Observations: ", dataframe.shape[0])
    print("Variables: ", dataframe.shape[1])
    print("Categorical Columns: ", len(cat_cols))
    print("Numeric columns: : ", len(num_cols))
    print("Cardinal Columns: ", len(cat_but_car))
    print("Num but categorical cols ", len(num_but_cat))

    return cat_cols, num_cols, cat_but_car


def target_summary_with_categorical(dataframe, target, cat_col):
    """
    This function prints the means of the target variable
    according to the classes of the given categorical variable

    Parameters
    ----------
    dataframe: pandas.DataFrame

    target: str
            the name of the target/dependent variable
    cat_col: str
            the name of the given categorical variable

    Returns
    -------
    """

    print(pd.DataFrame({"TARGET MEAN": dataframe.groupby(cat_col)[target].mean()}))


def target_summary_with_numeric(dataframe, target, num_col):
    """
    This function prints the means of the target variable
    according to the classes of the given numeric variable

    Parameters
    ----------
    dataframe: pandas.DataFrame

    target: str
            the name of the target/dependent variable
    num_col: str
            the name of the given numeric variable

    Returns
    -------
    """

    print(pd.DataFrame(dataframe.groupby(target).agg({num_col: "mean"})))


def high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    """
    This function returns the list of the high correlated variables.
    As default if the correlation between two variable bigger than 0.90
    this variables would be "high correlated".

    Parameters
    ----------
    dataframe: pandas.DataFrame

    plot: boolean
          If plot == True:
          plot bar plot of the categorical variable

    corr_th: int, float
             threshold value for high correlated variables

    Returns
    -------
    drop_list: list
               the list of high correlated variables
    """

    corr_ = dataframe.corr()
    corr_matrix_ = corr_.abs()
    upper_corr_ = corr_matrix_.where(np.triu(np.ones(corr_matrix_.shape), k=1).astype("bool"))
    drop_list = [col for col in upper_corr_.columns if any(upper_corr_[col] > corr_th)]

    if plot:

        sns.set(rc={"figure.figsize": (15, 25)})
        sns.heatmap(corr_, cmap="RdBu")
        plt.show()

    return drop_list
