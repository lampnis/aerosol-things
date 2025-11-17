# ------------------------ IMPORTS ----------------------------------

# og python libs
from typing import List, Dict, Any, Tuple, Pattern
import re
import os

# classic DS libs
import pandas as pd
import numpy as np

# viz libs
import matplotlib.pyplot as plt

# lmfit things
from lmfit.models import SplineModel

# ML
from sklearn.linear_model import LinearRegression

# GenAI
from google import genai
# from google.genai import types

# natural data databases
from my_utils.common import std_range, dir_match

# API setups
G_MODEL_ID = "gemini-2.5-flash-preview-05-20"
G_API_KEY = os.environ.get("GEMINI_API_KEY")
G_CLIENT = genai.Client(api_key=G_API_KEY)

col_names = {
    0: 'time',          # time (measurement tick, stopped on gasx, arbitrary)
    1: 'offset1',       # DFB offset-1
    2: 'P1',            # onboard pressure sensor (blue tube)
    3: 'H3',            # 3rd harmonic (L1H1)
    4: 'H2',            # 2nd harmonic (L1H2)
    5: 'P_las',         # laser power (some volts)
    6: 'T_las',         # diode temp (Kelvin)
    7: 'RH',            # humidity sensor (external)
    8: 'Pressure'       # sensor (external)
}


# ---------------------------- CLASSES -------------------------------
class CEPAS_measurement():

    def __init__(self, path: str, path_signature: str | Pattern[str],
                 cols: Dict[int, str]) -> None:
        """
        Initialize measurement from csv file to do data anlysis in
        python (pandas, numpy etc.)
        Collects all measurement files according to specified
        `path` and `path_signature`.
        collects all specified spectra in attribute `spectra_list`
        Adds `time_subtracted` column and `_pnorm` power normalized columns.
        Add 2 peaks of interest to see them up close in subplots.
        Can change peaks later with `redo_peaks()`

        Args:
            path (str): pass a path where spectra measurement files are located
            path_signature (str | Pattern[str]): pass a string or pattern to
            use as regex to find files
            cols (Dict[int, str]): pass a dict of columns to use as new column
            names

        Returns:
            None: but creates attributes
        """
        self.path = path
        self.raw_spectra = [
            pd.read_csv(
                path+self.dir_match(path_signature)[i],
                sep=r'\s+',
                header=None,
                names=list(cols.values()),
                index_col=False)
            for i in range(len(self.dir_match(path_signature)))
            ]
        self.n_spectra = len(self.raw_spectra)

        self.spectra_list: List[pd.DataFrame] = [
            self.raw_spectra[i].iloc[1:, :].copy()
            for i, _ in enumerate(self.raw_spectra)
            ]

        for df_idx in range(len(self.spectra_list)):

            self.spectra_list[
                df_idx
                ].loc[
                    :, 'time_subtracted'
                    ] = self.spectra_list[df_idx]['time']

            self.spectra_list[
                df_idx
                ].loc[
                    :, 'time_subtracted'
                    ] -= self.spectra_list[df_idx]['time'].min()

            self.spectra_list[
                df_idx
                ].loc[
                    :, 'H2_pnorm'
                    ] = self.spectra_list[
                        df_idx
                        ].loc[
                            :, 'H2'
                            ] / (self.spectra_list[
                                df_idx
                                ].loc[:, 'P_las']+0.0025)

            self.spectra_list[
                df_idx
                ].loc[
                    :, 'H3_pnorm'
                    ] = self.spectra_list[
                        df_idx
                        ].loc[
                            :, 'H3'
                            ] / (self.spectra_list[
                                df_idx
                                ].loc[
                                    :, 'P_las'
                                    ]+0.0025)

        self.mean_humidities = {
            df_idx: self.spectra_list[
                df_idx
                ]['RH'].mean() for df_idx in range(len(self.spectra_list))
        }
        try:
            self.mean_humidities_norm = np.array(
                list(
                    self.mean_humidities.values()
                )
                )/np.array(list(self.mean_humidities.values())).max()
        except ValueError as e:
            print(f"Low importance variable, replaced with none \
                  \n error: {e}")
            self.mean_humidities_norm = None

        self.peaks_starts_ends = {
            1: (40, 70),
            2: (150, 175)
        }

        # peak 1 idx 40-70
        self.peak1_start = self.peaks_starts_ends[1][0]
        self.peak1_end = self.peaks_starts_ends[1][1]

        self.peak_1s = [
            self.spectra_list[i].loc[
                self.peak1_start:self.peak1_end, :][
                    'H2'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_1s_norm = [
            self.spectra_list[
                i].loc[self.peak1_start:self.peak1_end, :][
                    'H2_pnorm'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_1s_norm_mins = [
            self.spectra_list[i].loc[
                self.peak1_start:self.peak1_end, :][
                    'H2_pnorm'].min() for i, _ in enumerate(self.spectra_list)]
        # print(f"self spectra list is {self.spectra_list}")
        # print(f"self.peak_1s_norm_mins is {self.peak_1s_norm_mins}")
        self.ylim_11 = np.array(self.peak_1s_norm_mins).min() - 0.1
        self.ylim_12 = np.array(self.peak_1s_norm).max() + 0.1

        # peak 2 idx 150-175
        self.peak2_start = self.peaks_starts_ends[2][0]
        self.peak2_end = self.peaks_starts_ends[2][1]

        self.peak_2s = [
            self.spectra_list[i].loc[
                self.peak2_start:self.peak2_end, :][
                    'H2'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_2s_norm = [
            self.spectra_list[i].loc[
                self.peak2_start:self.peak2_end, :][
                    'H2_pnorm'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_2s_norm_mins = [
            self.spectra_list[i].loc[
                self.peak2_start:self.peak2_end, :][
                    'H2_pnorm'].min() for i, _ in enumerate(self.spectra_list)]

        self.ylim_21 = np.array(self.peak_2s_norm_mins).min() - 0.1

        self.ylim_22 = np.array(self.peak_2s_norm).max() + 0.1

        self.pressures = [
            self.spectra_list[i]['Pressure'].unique()
            for i, _ in enumerate(self.spectra_list)]

    def replace(self, df_idx: int, df2: pd.DataFrame) -> None:
        """
        Replaces df1 w/ df2
        """
        self.spectra_list[df_idx] = df2

    def redo_peaks(self):
        """
        Redo the peaks according to whats been `set_` in `peak_bounds`
        """
        self.peak1_start = self.peaks_starts_ends[1][0]
        self.peak1_end = self.peaks_starts_ends[1][1]
        self.peak_1s = [
            self.spectra_list[i].loc[
                self.peak1_start:self.peak1_end, :][
                    'H2'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_1s_norm = [
            self.spectra_list[i].loc[
                self.peak1_start:self.peak1_end, :][
                    'H2_pnorm'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_1s_norm_mins = [
            self.spectra_list[i].loc[
                self.peak1_start:self.peak1_end, :][
                    'H2_pnorm'].min() for i, _ in enumerate(self.spectra_list)]

        self.ylim_11 = np.array(self.peak_1s_norm_mins).min() - 0.1
        self.ylim_12 = np.array(self.peak_1s_norm).max() + 0.1

        # peak 2 idx 150-175
        self.peak2_start = self.peaks_starts_ends[2][0]
        self.peak2_end = self.peaks_starts_ends[2][1]

        self.peak_2s = [
            self.spectra_list[i].loc[
                self.peak2_start:self.peak2_end, :][
                    'H2'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_2s_norm = [
            self.spectra_list[i].loc[
                self.peak2_start:self.peak2_end, :][
                    'H2_pnorm'].max() for i, _ in enumerate(self.spectra_list)]

        self.peak_2s_norm_mins = [
            self.spectra_list[i].loc[
                self.peak2_start:self.peak2_end, :][
                    'H2_pnorm'].min() for i, _ in enumerate(self.spectra_list)]

        self.ylim_21 = np.array(self.peak_2s_norm_mins).min() - 0.1
        self.ylim_22 = np.array(self.peak_2s_norm).max() + 0.1

    def set_peak_bounds(self, n_peak: int, start: int, end: int) -> None:
        """
        Set the peak parameters which will be used in visualization and
        in peak fitting (to get line height)

        Args:
            n_peak (int): which peak to set (1 or 2)
            start (int): peak start position (in offsets)
            end (int): peak end position (in offsets)

        Returns:
            None: but changes defaults in `peaks_starts_ends` dictionary
        """
        self.peaks_starts_ends[n_peak] = (start, end)

    def avg(self) -> pd.DataFrame:
        n = self.n_spectra
        if n == 0:
            raise ValueError("No spectra to average")
        spectra_sum = None
        for s in self.spectra_list:
            if spectra_sum is None:
                spectra_sum = s.copy()
            else:
                spectra_sum += s.copy()
        return spectra_sum / n  # type: ignore

    def water_plot(self,
                   save: bool = False,
                   save_path: str = "./water_plot_default.svg"
                   ) -> None:
        """
        Creates a plot with full spectrum,
        two peaks of choice,their intensities across measurements
        and normalized relative humidity across measurements

        Args:
            save (bool): save or not to save
            save_path: where to save
        Returns:
            None: but saves fig if specified!
        """
        plt.clf()
        fig = plt.figure(figsize=(17, 11))  # noqa: F841

        ax0 = plt.subplot2grid(shape=(12, 12),
                               loc=(6, 0),
                               colspan=6,
                               rowspan=6)
        for df_idx in range(len(self.spectra_list)):
            ax0.plot(self.spectra_list[df_idx]['offset1'],
                     self.spectra_list[df_idx]['H2_pnorm'],
                     label=f"{df_idx}")
        # ax0.set_xlim()
        # ax0.set_ylim(-0.5, 1)
        ax0.set_title("Power normalized spectrum")
        ax0.legend()
        ax0.set_xlabel("Arbitrary index")
        # ax0.set_xlabel("Laser current, arbitrary units")

        ax1 = plt.subplot2grid(shape=(12, 12),
                               loc=(0, 6),
                               colspan=5,
                               rowspan=4)
        if self.mean_humidities_norm is not None:
            ax1.scatter(list(self.mean_humidities.keys()),
                        self.mean_humidities_norm)
        ax1.set_xlabel('measurement #')
        ax1.set_ylabel('mean relative humidity, %')
        ax1.set_title("relative humidity over different measurement sessions")

        ax2 = plt.subplot2grid(shape=(12, 12),
                               loc=(4, 6),
                               colspan=5,
                               rowspan=4)
        plt.scatter(list(range(len(self.peak_1s_norm))),
                    self.peak_1s_norm)
        ax2.set_title("Peak 1 over sessions")
        ax2.set_xlabel("measurement #")
        ax2.set_ylabel("Normalized peak intensity")

        ax3 = plt.subplot2grid(shape=(12, 12),
                               loc=(8, 6),
                               colspan=5,
                               rowspan=4)
        ax3.scatter(list(range(len(self.peak_2s_norm))), self.peak_2s_norm)
        ax3.set_title("Peak 2 over sessions")
        ax3.set_xlabel("measurement #")
        ax3.set_ylabel("Normalized peak intensity")

        ax4 = plt.subplot2grid(shape=(12, 12),
                               loc=(0, 0),
                               colspan=3,
                               rowspan=6)
        for df_idx in range(len(self.spectra_list)):
            ax4.plot(self.spectra_list[df_idx]['offset1'],
                     self.spectra_list[df_idx]['H2_pnorm'],
                     label=f"{df_idx}")
        ax4.set_xlim(40, 75)  # x=index
        ax4.set_ylim(self.ylim_11, self.ylim_12)
        ax4.set_title("Peak 1")
        ax4.legend()
        # plt.xlabel("time, s")
        ax4.set_xlabel("arbitrary index")

        ax5 = plt.subplot2grid(shape=(12, 12),
                               loc=(0, 3),
                               colspan=3,
                               rowspan=6)
        for df_idx in range(len(self.spectra_list)):
            ax5.plot(self.spectra_list[df_idx]['offset1'],
                     self.spectra_list[df_idx]['H2_pnorm'],
                     label=f"{df_idx}")
        ax5.set_xlim(142, 182)  # x=index
        ax5.set_ylim(self.ylim_21, self.ylim_22)
        ax5.set_title("Peak 2")
        ax5.legend()
        # plt.xlabel("time, s")
        ax4.set_xlabel("arbitrary index")

        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{save_path}")

    def spectrum_only_plot(self,
                           save: bool = False,
                           save_path: str = "./spectrum_only_plot_default.svg"
                           ) -> None:
        """
        Makes just a simple plot of spectrum with all measurements
        laid over each other

        Args:
            save (bool): save or not to save
            save_path: where to save
        Returns:
            None: but saves fig if specified!
        """
        plt.close()
        fig = plt.figure(figsize=(17, 5.5))  # noqa: F841

        ax0 = plt.subplot2grid(shape=(12, 12), loc=(0, 0),
                               colspan=12,
                               rowspan=12)
        for df_idx in range(len(self.spectra_list)):
            ax0.plot(self.spectra_list[df_idx]['offset1'],
                     self.spectra_list[df_idx]['H2_pnorm'],
                     label=f"{df_idx}")
        # ax0.set_xlim()
        # ax0.set_ylim(-0.5, 1)
        ax0.set_title("Power normalized spectrum")
        ax0.legend()
        ax0.set_xlabel("Arbitrary index")

        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{save_path}")

    def dir_match(self, pattern: str | Pattern[str]) -> List[str]:
        """
        Iterates through contents of dir and matches according to some regex \
            pattern

        Args:
            path (str): path to directory of interest
            pattern (str): pattern to find, either simple or regex

        Returns:
            list[str]: All matches
        """
        matched = []
        dir_contents = sorted(os.listdir(self.path))
        for item in dir_contents:
            if re.match(pattern, item) is not None:
                matched.append(item)
        return matched

    def add_wavenumber_axis(self, df, units1, units2):
        """
        adds a wavenumber axis based on `get_wavenumber()` function
        """
        df['wavenumbers'] = self.get_wavenumber(df['offset1'],
                                                units1,
                                                units2)[0]

    def get_wavenumber(self,
                       unit1: float | pd.Series,
                       units1: List[float],
                       units2: List[float]
                       ) -> List[float | pd.Series]:
        """
        Converts `unit1` to `unit2` based on the
        `sklearn.linear_model.LinearRegression()`
        fit of lists `units1` and `units2`

        Args:
            unit1 (float): Value of interest (unit from which will convert)
            units1 (list[float]): List of values in units of interest
            units2 (list[float]): List of values in units from which we are \
                converting

        Returns:
            List[float]: Value in units of interest, slope, intercept
        """
        lr_wavenumbers = LinearRegression()
        offset_arr = np.array(units1).reshape(-1, 1)
        wavnumber_arr = np.array(units2).reshape(-1, 1)
        lr_wavenumbers.fit(offset_arr, wavnumber_arr)
        a = float(lr_wavenumbers.coef_[0][0])
        b = float(lr_wavenumbers.intercept_[0])  # type: ignore
        return [a*unit1 + b, a, b]


class CEPAS_benchmark():
    def __init__(self,
                 path: str,
                 spectra_names: Dict[
                     int | str | float, Dict[
                         int | str | float, List[str]]],
                 pressure: int | str | float,
                 frequency: int | str | float,
                 noise_flag: bool = False,
                 file_signature: str = "gasx") -> None:
        """
        Create the object for benchmark measurements.
        Keeps everything organized in the dictionary (depth=2).
        Based on `CEPAS_measurement()` class.

        Args:
            path (str): path to the benchmark spectral measurements
            spectra_names (Dict[int | str | float, Dict[int | str | float, \
                List[str]]]): \
                special dictionary that stores the filenames of the spectra
            pressure (int|str|float): pressure to look at
            frequency (int|str|float): frequency to look at
            noise_flag (bool): take care of the triplets that are \
                not the same in length (align x axis)

        Returns:
            None: but initializes the class
        """
        self.path = path
        self.spectra_names = spectra_names[pressure][frequency]
        self.pressure = pressure
        self.frequency = frequency
        self.spectra = CEPAS_measurement(
            path=path,
            path_signature=f"{
                file_signature}_{
                    pressure}_[0-9]{{2}}_{
                        frequency}_.+",
            cols=col_names)
        self.noise_flag = noise_flag

        # workaround for uneven spectra
        if not self.noise_flag and len(self.spectra.spectra_list) > 1:
            neq_12 = len(self.spectra.spectra_list[0]) != \
                len(self.spectra.spectra_list[1])
            neq_23 = len(self.spectra.spectra_list[1]) != \
                len(self.spectra.spectra_list[2])
            # neq_13 = len(self.spectra.spectra_list[2]) != \
            # len(self.spectra.spectra_list[0])

            if neq_12:
                self.spectra.replace(
                    1,
                    self.spectra.spectra_list[1].iloc[1:].reset_index(
                        drop=True))
                self.spectra.replace(
                    2,
                    self.spectra.spectra_list[2].iloc[1:].reset_index(
                        drop=True))
                self.spectra.replace(
                    0,
                    self.spectra.spectra_list[0].reset_index(
                        drop=True))

            if neq_23:
                self.spectra.replace(
                    0,
                    self.spectra.spectra_list[0].iloc[1:].reset_index(
                        drop=True))
                self.spectra.replace(
                    1,
                    self.spectra.spectra_list[1].iloc[1:].reset_index(
                        drop=True))
                self.spectra.replace(
                    2,
                    self.spectra.spectra_list[2].reset_index(
                        drop=True))

            # print(self.spectra.spectra_list)

    def add_magnitude(self):
        """
        Takes `H2_pnorm` and `H3_pnorm` colums \
        and returns root of quadrature of those. \
        Think of it as a magnitude of complex number
        """
        for s in self.spectra.spectra_list:
            s["magnitude_pnorm"] = np.sqrt(s["H2_pnorm"]**2 + s["H3_pnorm"]**2)

    def get_avg(self) -> pd.DataFrame:
        """
        Get the avg of three benchmark spectra
        """
        return self.spectra.avg()

    def add_wav(self,
                units1: List[float | int],
                units2: List[float | int]) -> None:
        """
        Adds wavenumber axis, based on the \
            `CEPAS_measurement.get_wavenumber()`.
        Does a linear regression of two units \
            to determine the conversion factors.

        Args:
            units1 (List[float | int]): unit converted from \
                (in this application usually `offset1`)
            units2 (List[float | int]): unit to convert to \
                (here usually wavenumbers)

        Returns:
            None
        """
        for s in self.spectra.spectra_list:
            s['wavenumbers'] = self.spectra.get_wavenumber(
                s['offset1'],
                units1,
                units2
                )[0]

    def self_test(self) -> None:
        """
        Does a small functional test for the class
        """
        test_string = f"""\
        ------------Benchmark test-------------
        Path: {self.path}
        File names:\n\t{self.spectra_names}
        pressure: {self.pressure} mbar
        frequency: {self.frequency} Hz
        spectra_list: {self.spectra.spectra_list}
        ----------------END--------------------
        """
        print(test_string)

    def get_window(self,
                   start: int | float = 1625,
                   end: int | float = 1750,
                   col: str = 'offset1') -> List[pd.DataFrame]:
        """
        Gets the region of the spectrum.

        Args:
            start (int | float): start of the region, in offsets
            end (int | float): end of the region, in offsets
            col (str): column which contains the values of start and end

        Returns:
            List[pd.DataFrame]: Tha last one is the average \
                of all the previous ones
        """
        frames = self.spectra.spectra_list + [self.get_avg()]
        cut_frames = [f[(f[col] > start) & (f[col] < end)] for f in frames]

        return cut_frames

    def get_spline_of_window(self,
                             n_spectrum: int = -1,
                             n_knots: int = 10,
                             start: int | float = 1625,
                             end: int | float = 1750,
                             colx: str = 'offset1',
                             coly: str = 'H2_pnorm',
                             n_dense: int = 100) -> Tuple[
                                 pd.DataFrame,
                                 str,
                                 pd.DataFrame,
                                 pd.DataFrame,
                                 float
                                 ]:
        """
        Does the `lmfit.SplineModel` on defined range of data

        Args:
            n_spectrum (int): Which spectrum from the benchmark to analyse. \
                Defaults to -1, which is the average of all measurements
            n_knots (int): number of knots used in `lmfit.SplineModel()`
            start (int | float): Start of the window of interest
            end (int | float): End of the window of interest
            colx (str): Which column to use for x-axis, default is `'offset1'`
            coly (str): Which column to use for y-axis, default is `'H2_pnorm'`
            n_dense (int): How many points to use in the \
                evaluation of the spline, \
                    using the x-axis of arbitrary \
                        density, to get more accurate max value

        Returns:
            (Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame, float]): \
                best fit, fit report, dense evaluation and row of \
                    dense evaluation at `coly=coly.max()`, \
                        finally a max of coly float itself
        """

        buffer = 0.01
        avg = self.get_window(start, end, colx)[n_spectrum]
        x = avg[colx].to_numpy()
        y = avg[coly].to_numpy()
        knot_xvals = np.linspace(start+buffer, end-buffer, n_knots)
        model = SplineModel(xknots=knot_xvals)
        params = model.guess(y, x=x)
        result = model.fit(y, params, x=x)
        x_dense = np.linspace(start+buffer, end-buffer, n_dense)
        result_df = pd.DataFrame(
            {
                colx: x,
                coly: result.best_fit
            }
        )
        result_dense = pd.DataFrame(
            {
                colx: x_dense,
                coly: result.eval(x=x_dense)
            }
        )
        peak_height = result_dense[result_dense[coly] ==
                                   result_dense[coly].max()]

        return result_df, result.fit_report(), result_dense, peak_height, \
            peak_height[coly]


class CEPAS_noise_info():

    def __init__(self,
                 path: str,
                 pressure: int | str | float='',
                 n: int | None = None):
        """
        Initiate the noise info. If there are multiple \
        sessions where the noise could be different, \
        it should be indicated with `n`

        Args:
            path (str): where the noise spectra are located
            pressure (int | str | float): which pressure, \
                default is an empty string
            n (int): optional, choose specific session of \
            noise spectra measrement
        """

        if n is not None:

            re_str = dir_match(
                path,
                re.compile(f"Spectrum.*{pressure}_{n}.txt")
            )

            self.spectrum = pd.read_csv(
                f"{path}{re_str[0]}",
                sep=r'\s+',
                header=None,
                names=['freq',
                       'intensity'])
        else:

            re_str = dir_match(
                path,
                re.compile(f"Spectrum.*{pressure}.txt")
            )
            self.spectrum = pd.read_csv(
                f"{path}{re_str[0]}",
                sep=r'\s+',
                header=None,
                names=['freq',
                       'intensity'])

    def get_noise_at(self,
                     frequency: int | str | float,
                     buffer: int | float = 2,
                     h: int = 2) -> float:
        """
        Get the system background at the specified frequency.
        Pressure is set upon making initiating the object.

        Args:
            frequency (int | str | float): At what frequency you want to get \
                system background signal?
            buffer (int | float): What should \
                be the averaging range (+/- buffer)
            h (int | None): What is the harmonic used \
                for signal measurements (default=2)

        Returns:
            (float): gives some value at
        """
        # Ensure frequency is a float for arithmetic
        try:
            frequency = float(frequency)*h
        except Exception:
            raise ValueError(
                f"Frequency must be convertible to float, \
                    got {frequency} of type {type(frequency)}")

        start = frequency - buffer
        end = frequency + buffer
        df = self.spectrum
        return df[
            (df['freq'] > start) & (df['freq'] < end)
            ]['intensity'].mean()


class CEPAS_SNR_bench():

    def __init__(self,
                 snr_dict: Dict[
                     int | str | float,
                     Dict[
                         int | str | float,
                         List[str]]],
                 bench_path: str = "",
                 noise_path: str = "",
                 noise_number: None | int = None,
                 file_sig: str = "gasx"):
        self.snr_dict = snr_dict
        self.bench_path = bench_path
        self.bench_files = snr_dict
        self.noise_path = noise_path
        self.noise_number = noise_number
        self.file_sig = file_sig
    
    def change_file_sig(self, new_file_sig: str) -> None:
        """
        changes the `file_sig` if necessary
        """
        self.file_sig = new_file_sig

    def set_noise_path(self, path: str) -> None:
        """
        Sets a path in case it was not set when creating the object
        """
        self.noise_path = path

    def set_bench_path(self, path: str) -> None:
        """
        Sets a path in case it was not set when creating the object
        """
        self.bench_path = path

    def make_bench(self,
                   pressure: int | str | float,
                   frequency: int | str | float,
                   file_sig: str = 'gasx') -> CEPAS_benchmark: # type: ignore # noqa: F821
        """
        Create a `CEPAS_benchmark` object for the chemical spectra
        """
        if self.bench_path == "":
            raise ValueError("Please \
                             create a bench path using \
                             'object.set_bench_path(path)'")
        try:
            bench_test = CEPAS_benchmark(
                self.bench_path,
                self.bench_files,
                pressure, frequency,
                file_signature=self.file_sig
                )
        except ValueError as e:
            print("Wrong filename, changed:")
            print(f"'single'->'gasx', {e}")
            bench_test = CEPAS_benchmark(
                self.bench_path,
                self.bench_files,
                pressure, frequency,
                file_signature=file_sig
                )
        return bench_test

    def make_noise_bench(self,
                         pressure: int | str | float,
                         frequency: int | str | float) -> CEPAS_benchmark:
        """
        Create a `CEPAS_benchmark` object for the noise of single points
        """
        if self.noise_path == "":
            raise ValueError(
                "Please create a noise \
                    path using 'object.set_noise_path(path)'")
        bench_noise = CEPAS_benchmark(
            self.noise_path,
            self.bench_files,
            pressure,
            frequency,
            noise_flag=True,
            file_signature=self.file_sig
            )
        return bench_noise

    def get_signal(self,
                   start: int | float,
                   end: int | float,
                   pressure: int | str | float,
                   frequency: int | str | float,
                   n_knots: int = 11,
                   coly: str = "H2_pnorm",) -> Tuple[
                       pd.DataFrame,
                       str,
                       pd.DataFrame,
                       pd.DataFrame,
                       float
                       ]:
        """
        Returns the chosen signal peak in spectral region `start` to `end`.

        Args:
            start (int|float): start of the peak window
            end (int|float): end of the peak window
            pressure (int|str|float): pressure of interest
            frequency (int|str|float): frequency of interest
            n_knots (int): default = 11, for backend splines, adjust \
                if necessary

        Returns:
            (Tuple[pd.DataFrame, str, pd.DataFrame, pd.Dataframe, float]): \
                signal level is the last one. For other values, see \
                `CEPAS_benchmark.get_spline_of_window()` docstring

        """
        try:
            bench = self.make_bench(pressure, frequency, file_sig="gasx")
            peak_start = start
            peak_end = end
            peak_spline = bench.get_spline_of_window(
                n_spectrum=-1,
                n_knots=n_knots,
                start=peak_start,
                end=peak_end,
                coly=coly
                )
            return peak_spline

        except KeyError as e:
            print(f"{e}\n, but now added the missing column")
            bench = self.make_bench(pressure, frequency)
            bench.add_magnitude()
            peak_start = start
            peak_end = end
            peak_spline = bench.get_spline_of_window(
                n_spectrum=-1,
                n_knots=n_knots,
                start=peak_start,
                end=peak_end,
                coly=coly
                )
            return peak_spline

    def get_horizontal_noise(self,
                             start: int | float,
                             end: int | float,
                             pressure: int | str | float,
                             frequency: int | str | float,
                             wstart: int | float = 1687.5,
                             wend: int | float = 1707.5,
                             n_knots: int = 10):
        """
        Get the horizontal noise at some window
        """
        bench = self.make_bench(pressure, frequency)
        flat_regions = bench.get_window(start=start, end=end)
        spline_of_avg = bench.get_spline_of_window(
            n_knots=n_knots,
            start=start,
            end=end)
        flat_regions_dict = {
            '0': flat_regions[0],
            '1': flat_regions[1],
            '2': flat_regions[2],
            'avg': flat_regions[3],
            'avg_spline': spline_of_avg[0]
        }

        stdevs_horizontal = []
        for k in flat_regions_dict.keys():
            df = flat_regions_dict[k]
            flat_regions_dict[k] = df[
                (df['offset1'] > wstart) & (df['offset1'] < wend)]
            df = flat_regions_dict[k]
            stdevs_horizontal.append(df['H2_pnorm'].std(ddof=0))

        all_horizontal_noises = [
            np.format_float_scientific(
                i,
                precision=3
                ) for i in stdevs_horizontal]
        horizontal_mean = np.mean(np.array(stdevs_horizontal[0:3]))
        return all_horizontal_noises, horizontal_mean

    def get_vertical_noise(self,
                           start: int | float,
                           end: int | float,
                           pressure: int | str | float,
                           frequency: int | str | float,
                           wstart: int | float = 1687.5,
                           wend: int | float = 1707.5,
                           n_knots: int = 10) -> Tuple[
                               List[str] | List[float],
                               float | np.floating[Any]
                               ]:
        """
        Get the vertical noise at some predefined window
        """
        bench = self.make_bench(pressure, frequency)
        flat_regions = bench.get_window(start=start, end=end)
        spline_of_avg = bench.get_spline_of_window(
            n_knots=n_knots,
            start=start,
            end=end
            )
        flat_regions_dict = {
            '0': flat_regions[0],
            '1': flat_regions[1],
            '2': flat_regions[2],
            'avg': flat_regions[3],
            'avg_spline': spline_of_avg[0]
        }

        for k in flat_regions_dict.keys():
            df = flat_regions_dict[k]
            flat_regions_dict[k] = df[
                (df['offset1'] > wstart) & (df['offset1'] < wend)
                ]

        stdevs_vertical = []
        df_std = pd.DataFrame(
            {k: v.reset_index()['H2_pnorm']
             for k, v in flat_regions_dict.items()})
        df_std = df_std.loc[:, '0':'2']
        for i in range(4):
            stdevs_vertical.append(df_std.iloc[i].T.std(ddof=0))

        all_vertical_noises = [
            np.format_float_scientific(
                i,
                precision=3
                ) for i in stdevs_vertical
                ]
        vertical_mean = np.average(np.array(stdevs_vertical[0:4]))
        return all_vertical_noises, vertical_mean

    def get_single_point_noise(self,
                               pressure: int | str | float,
                               frequency: int | str | float) -> Tuple[
                                   pd.DataFrame,
                                   float,
                                   pd.DataFrame]:
        """
        Wrapper for `std_range()` function,
        applied to single point noise spectrum `df`

        Args:
            pressure (int|str|float): what pressure \
            frequency (int|str|float): what float

        Returns:
            (Tuple[pd.DataFrame, float, pd.DataFrame]): See \
            the definition of `std_range()`
        """
        bench = self.make_noise_bench(pressure, frequency)
        df = bench.spectra.spectra_list[0]
        y = 'H2'
        n = 3
        return std_range(df, y, n)

    def get_clean_noise(
            self,
            pressure: int | str | float,
            frequency: int | str | float) -> Tuple[
                None, float
                ]:
        """
        Get the clean noise from noise spectra data, by taking the mean \
        around some central frequency

        Args:
            pressure (int | str | float): which pressure noise to take
            frequency (int | str | float): which frequency to choose \
                as central frequency

        Returns:
            (Tuple[None, float]): get the float number. \
            For consistency (index=1), 0th index is None
        """
        if self.noise_path == "":
            raise ValueError(
                "Please create a noise path \
                    using 'object.set_noise_path(path)'")
        return None, CEPAS_noise_info(
            path=self.noise_path,
            pressure=pressure,
            n=self.noise_number).get_noise_at(frequency=frequency)

    def get_all_snrs(self,
                     
                     start: List[int | float],
                     end: List[int | float],
                     h: int = 2,
                     n_knots: int = 10) -> Tuple[
                         pd.DataFrame,
                         pd.DataFrame
                         ]:
        """
        Creates the final report for barplots of signal/noise/SNR

        Args:
            start (List[int|float]): List of two items, peak start and noise \
            start
            end (List[int|float]): List of two items, peak end and noise end
            h (int): harmonic used
            n_knots (int): for backend splines, adjust if necessary, default=10

        Returns:
            (Tuple[pd.DataFrame, pd.DataFrame]): df based on \
                SNR dict and noise dict, \
                which organizes final data by pressure and modulation frequency
        """
        snr_dict = {}
        noise_dict = {}
        for p in self.snr_dict.keys():
            snr_dict[p] = {}
            noise_dict[p] = {}
            for f in self.snr_dict[p].keys():
                snr_dict[p][f*h] = []
                noise_dict[p][f*h] = []
                signal = self.get_signal(start[0],
                                         end[0],
                                         p, f,
                                         n_knots=n_knots)
                # noise_h = self.get_horizontal_noise(start[1], end[1], p, f)
                # noise_v = self.get_vertical_noise(start[1], end[1], p, f)
                noise_h = self.get_horizontal_noise(
                    start[1],
                    end[1], p, f,
                    wstart=1657.5,
                    wend=1677.5,
                    n_knots=n_knots)
                noise_v = self.get_vertical_noise(
                    start[1],
                    end[1], p, f,
                    wstart=1657.5,
                    wend=1677.5,
                    n_knots=n_knots)
                if p == 900:
                    noise_h = self.get_horizontal_noise(start[1], end[1], p, f,
                                                        n_knots=n_knots)
                    noise_v = self.get_vertical_noise(start[1], end[1], p, f,
                                                      n_knots=n_knots)
                if p == 600:
                    noise_h = self.get_horizontal_noise(
                        start[1],
                        end[1], p, f,
                        wstart=1682.5,
                        wend=1702.5,
                        n_knots=n_knots)
                    noise_v = self.get_vertical_noise(
                        start[1],
                        end[1], p, f,
                        wstart=1682.5,
                        wend=1702.5,
                        n_knots=n_knots)
                noise_c = self.get_clean_noise(p, f)
                noise_s = self.get_single_point_noise(p, f)
                snr_h = signal[-1] / noise_h[1]
                snr_v = signal[-1] / noise_v[1]
                snr_s = signal[-1] / noise_s[1]
                snr_c = signal[-1] / noise_c[1]

                print(
                    f"DEBUG: At p={p} \
                        and f={f} signal is \n---->{signal[-1]}<----\n")
                print(f"noise from single measurements: {noise_s[1]}")

                for snr in [snr_h, snr_v, snr_s, snr_c]:
                    snr_dict[p][f*h].append(float(snr.iloc[0]))
                for n in [noise_h, noise_v, noise_s, noise_c]:
                    noise_dict[p][f*h].append(float(n[1]))

        return pd.DataFrame(snr_dict), pd.DataFrame(noise_dict)

    def get_mag_snrs(self,
                     start: int | float,
                     end: int | float,
                     n_knots: int = 10,
                     h: int = 2,
                     signal_col: str = "magnitude_pnorm",
                     skip_single: bool = False) -> Tuple[
                         pd.DataFrame,
                         pd.DataFrame,
                         pd.DataFrame
                         ]:
        """
        Creates the final report for barplots of signal/noise/SNR \
        for magnitude spectra

        Args:
            start (int|float): peak start \
            start
            end (int|float): peak end
            n_knots (int): for backend splines, adjust if necessary, default=10
            h (int): harmonic used
            signal_col (str): usually magnitude here
            skip_single (bool): default false, toggle if there are no single point \
                measurements

        Returns:
            (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): df based on \
                SNR dict and noise dict, and signal dict \
                which organizes final data by pressure and modulation frequency
        """
        snr_dict = {}
        noise_dict = {}
        signal_dict = {}

        for p in self.snr_dict.keys():
            snr_dict[p] = {}
            noise_dict[p] = {}
            signal_dict[p] = {}
            for f in self.snr_dict[p].keys():
                snr_dict[p][f*h] = []
                noise_dict[p][f*h] = []
                signal_dict[p][f*h] = []
                signal = self.get_signal(start,
                                         end,
                                         p, f,
                                         n_knots=n_knots,
                                         coly=signal_col)

                try:
                    noise_c = self.get_clean_noise(p, f)
                except IndexError as e:
                    noise_c = self.get_clean_noise('', f)
                    print(f"Replaced pressure in clean noise file \
                          name with empty string, {e}")
                snr_c = signal[-1] / noise_c[1]
                
                # ensure variables are always defined to satisfy static analysis
                noise_s = None
                snr_s = None
                if not skip_single:
                    noise_s = self.get_single_point_noise(p, f)
                    snr_s = signal[-1] / noise_s[1]

                print(
                    f"DEBUG: At p={p} \
                        and f={f} signal is \n---->{signal[-1]}<----\n")
                if not skip_single and noise_s is not None:
                    print(f"noise from single measurements: {noise_s[1]}")
                else:
                    print("!No single point measurements this time!")

                if not skip_single:
                    for snr in [snr_s, snr_c]:
                        snr_dict[p][f*h].append(float(snr.iloc[0]))  # type: ignore
                    for n in [noise_s, noise_c]:
                        noise_dict[p][f*h].append(float(n[1]))
                    signal_dict[p][f*h].append(signal[-1])
                else:
                    for snr in [snr_c]:
                        snr_dict[p][f*h].append(float(snr.iloc[0]))  # type: ignore
                    for n in [noise_c]:
                        noise_dict[p][f*h].append(float(n[1]))
                    signal_dict[p][f*h].append(signal[-1])

        return (pd.DataFrame(snr_dict),
                pd.DataFrame(noise_dict),
                pd.DataFrame(signal_dict))
