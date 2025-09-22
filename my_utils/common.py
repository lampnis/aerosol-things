import pandas as pd
from typing import Tuple


def std_range(
    df: pd.DataFrame,
    y: str,
    coef: int | float = 1
) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Filters data. Creates new stddev for some filtered data based on the IQR
    from 0.25 quantile to 0.75 quantile.

    Args:
        df (pd.DataFrame): pandas dataframe to work on
        y (str): column name of the column of interest
        coef (int | float): larger number makes filtering weaker (default=1)

    Returns:
        (Tuple[pd.DataFrame, float, pd.DataFrame]): new data frame, new stddev, outliers
    """
    Q1 = df[y].quantile(0.25)
    Q3 = df[y].quantile(0.75)
    IQR = Q3 - Q1
    new_df = df[(df[y] > Q1-coef*IQR) & (df[y] < Q3+coef*IQR)]
    outliers = df[(df[y] < Q1-coef*IQR) & (df[y] > Q3+coef*IQR)]
    new_std = new_df[y].std()
    return new_df, new_std, outliers
