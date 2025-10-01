import pandas as pd
from typing import Tuple, List, Dict, Pattern, Any
from classes import CEPAS_benchmark
import defs as cp
import matplotlib.pyplot as plt
import numpy as np


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
        (Tuple[pd.DataFrame, float, pd.DataFrame]): new data frame, \
            new stddev, outliers
    """
    Q1 = df[y].quantile(0.25)
    Q3 = df[y].quantile(0.75)
    IQR = Q3 - Q1
    new_df = df[(df[y] > Q1-coef*IQR) & (df[y] < Q3+coef*IQR)]
    outliers = df[(df[y] < Q1-coef*IQR) & (df[y] > Q3+coef*IQR)]
    new_std = new_df[y].std()
    return new_df, new_std, outliers


def plot_magnitudes(path: str,
                    patterns: Dict[
                        int | str | float,
                        Dict[
                            int | str | float,
                            Pattern[str]
                        ]
                    ],
                    p: List[int],
                    freqs: List[int]) -> Dict[Any, Any]:
    """
    Plot the magnitudes for phase uncorrected spectra

    Args:
        path (str): Where are the spectra located
        patterns (Dict[int | str | float, \
            Dict[int | str | float, Pattern[str]]): \
            patterns built with `dir_match_dict()`
        p (List[int]): pressures list
        freqs (List[int]): frequencies list

    Returns:
        (Dict[Any, Any]): Benches (Plots the magnitude spectra of these)

    """

    dictionary = cp.dir_match_dict(path, patterns)
    to_return = {}

    for pressure in p:
        benches = [
            CEPAS_benchmark(path, dictionary, pressure, f) for f in freqs
        ]

        # add averages
        path_avgs = [bench.get_avg() for bench in benches]

        for bench_idx in range(len(benches)):
            spectra = benches[bench_idx].spectra.spectra_list
            spectra.append(path_avgs[bench_idx])

        # plot everything
        plt.close()
        for b_idx in range(len(benches)):
            spectra = benches[b_idx].spectra.spectra_list
            plt.figure()
            labels = ['1', '2', '3', 'avg']
            for s_idx in range(len(spectra)):
                magnitude = np.sqrt(
                    spectra[s_idx]['h2_pnorm']**2 +
                    spectra[s_idx]['h3_pnorm']**2
                )
                plt.plot(
                    spectra[s_idx]['offset1'],
                    magnitude,
                    label=labels[s_idx]
                )
            plt.title(f"$f={freqs[b_idx]}$")
            plt.legend()
            plt.show()

        to_return[p] = benches

    return to_return
