# `my_utils/` - Aerosol Data Analysis Utilities

This directory contains a collection of Python modules designed to assist with the analysis, processing, and visualization of aerosol measurement data. These utilities are used across various Jupyter notebooks in the project to ensure consistency and reduce code duplication.

## Module Overview

*   **`classes.py`**: Defines core data structures and classes for handling experimental measurements, benchmarks, noise analysis, and Signal-to-Noise Ratio (SNR) calculations.
*   **`common.py`**: Provides general-purpose utility functions for data manipulation, such as filtering and file matching.
*   **`defs.py`**: Contains project-wide constants, plotting helpers, and specialized functions for spectroscopic analysis, including interactions with the HITRAN database and Voigt profile fitting.

---

## Detailed Documentation

### `my_utils/classes.py`

This module encapsulates the main data handling logic through several classes.

#### `CEPAS_measurement` Class

Initializes and processes raw CSV measurement files, providing structured access to spectral data, humidity information, and peak characteristics.

**Attributes:**
*   `path` (str): Path to the directory containing measurement files.
*   `raw_spectra` (List[pd.DataFrame]): List of raw spectra DataFrames.
*   `n_spectra` (int): Number of spectra loaded.
*   `spectra_list` (List[pd.DataFrame]): List of processed spectra DataFrames with added `time_subtracted`, `H2_pnorm`, and `H3_pnorm` columns.
*   `mean_humidities` (Dict[int, float]): Mean humidity for each spectrum.
*   `mean_humidities_norm` (np.ndarray | None): Normalized mean humidities.
*   `peaks_starts_ends` (Dict[int, Tuple[int, int]]): Dictionary storing start and end indices for defined peaks.
*   `peak1_start`, `peak1_end`, `peak2_start`, `peak2_end` (int): Start and end indices for peak 1 and peak 2.
*   `peak_1s`, `peak_2s` (List[float]): Maximum H2 intensity for peak 1 and peak 2.
*   `peak_1s_norm`, `peak_2s_norm` (List[float]): Maximum power-normalized H2 intensity for peak 1 and peak 2.
*   `peak_1s_norm_mins`, `peak_2s_norm_mins` (List[float]): Minimum power-normalized H2 intensity for peak 1 and peak 2.
*   `ylim_11`, `ylim_12`, `ylim_21`, `ylim_22` (float): Y-axis limits for plotting peaks.
*   `pressures` (List[np.ndarray]): Unique pressure values for each spectrum.

**Methods:**
*   `__init__(self, path: str, path_signature: str | Pattern[str], cols: Dict[int, str]) -> None`:
    Initializes the `CEPAS_measurement` object.
    *   `path`: Directory where spectra measurement files are located.
    *   `path_signature`: Regex pattern to find files.
    *   `cols`: Dictionary mapping column indices to new column names.
*   `replace(self, df_idx: int, df2: pd.DataFrame) -> None`: Replaces a DataFrame at a specific index in `spectra_list`.
*   `redo_peaks(self) -> None`: Recalculates peak attributes based on current `peaks_starts_ends`.
*   `set_peak_bounds(self, n_peak: int, start: int, end: int) -> None`: Sets the start and end indices for a specified peak.
*   `avg(self) -> pd.DataFrame`: Returns the average of all spectra in `spectra_list`.
*   `water_plot(self, save: bool = False, save_path: str = "./water_plot_default.svg") -> None`: Generates a comprehensive plot including full spectrum, two selected peaks, their intensities across measurements, and normalized relative humidity.
*   `spectrum_only_plot(self, save: bool = False, save_path: str = "./spectrum_only_plot_default.svg") -> None`: Creates a simple plot of all spectra overlaid.
*   `dir_match(self, pattern: str | Pattern[str]) -> List[str]`: Iterates through a directory and returns files matching a regex pattern.
*   `add_wavenumber_axis(self, df, units1, units2)`: Adds a 'wavenumbers' column to a DataFrame based on a linear regression conversion.
*   `get_wavenumber(self, unit1: float | pd.Series, units1: List[float], units2: List[float]) -> List[float | pd.Series]`: Converts `unit1` to `unit2` using linear regression.

#### `CEPAS_benchmark` Class

Manages benchmark measurements, organizing spectra by pressure and frequency, and providing methods for analysis.

**Attributes:**
*   `path` (str): Path to the benchmark spectral measurements.
*   `spectra_names` (List[str]): List of filenames for the current benchmark.
*   `pressure` (int | str | float): Pressure value for the benchmark.
*   `frequency` (int | str | float): Frequency value for the benchmark.
*   `spectra` (CEPAS_measurement): A `CEPAS_measurement` object containing the benchmark spectra.
*   `noise_flag` (bool): Indicates if the benchmark is for noise measurements (handles uneven spectra lengths).

**Methods:**
*   `__init__(self, path: str, spectra_names: Dict[int | str | float, Dict[int | str | float, List[str]]], pressure: int | str | float, frequency: int | str | float, noise_flag: bool = False, file_signature: str = "gasx") -> None`:
    Initializes the `CEPAS_benchmark` object.
    *   `path`: Path to the benchmark spectral measurements.
    *   `spectra_names`: Dictionary storing filenames of spectra.
    *   `pressure`: Pressure to look at.
    *   `frequency`: Frequency to look at.
    *   `noise_flag`: If True, handles uneven spectra lengths (for noise measurements).
    *   `file_signature`: Prefix for filenames (e.g., "gasx").
*   `add_magnitude(self) -> None`: Adds a `magnitude_pnorm` column (root of quadrature of `H2_pnorm` and `H3_pnorm`).
*   `get_avg(self) -> pd.DataFrame`: Returns the average of the benchmark spectra.
*   `add_wav(self, units1: List[float | int], units2: List[float | int]) -> None`: Adds a 'wavenumbers' axis to all spectra in the benchmark.
*   `self_test(self) -> None`: Prints a functional test string for the class.
*   `get_window(self, start: int | float = 1625, end: int | float = 1750, col: str = 'offset1') -> List[pd.DataFrame]`: Returns a list of DataFrames representing a specific region of the spectrum.
*   `get_spline_of_window(self, n_spectrum: int = -1, n_knots: int = 10, start: int | float = 1625, end: int | float = 1750, colx: str = 'offset1', coly: str = 'H2_pnorm', n_dense: int = 100) -> Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame, float]`: Performs `lmfit.SplineModel` on a defined range of data and returns fit results.

#### `CEPAS_noise_info` Class

Handles and extracts information from noise spectra data.

**Attributes:**
*   `spectrum` (pd.DataFrame): DataFrame containing the noise spectrum.

**Methods:**
*   `__init__(self, path: str, pressure: int | str | float = '', n: int | None = None) -> None`:
    Initializes the `CEPAS_noise_info` object.
    *   `path`: Directory where noise spectra are located.
    *   `pressure`: Pressure for the noise spectrum (optional).
    *   `n`: Specific session number for noise spectra (optional).
*   `get_noise_at(self, frequency: int | str | float, buffer: int | float = 2, h: int = 2) -> float`: Returns the mean system background noise at a specified frequency.

#### `CEPAS_SNR_bench` Class

Calculates Signal-to-Noise Ratios (SNR) by integrating `CEPAS_benchmark` and `CEPAS_noise_info`.

**Attributes:**
*   `snr_dict` (Dict): Dictionary storing SNR data.
*   `bench_path` (str): Path to benchmark spectral measurements.
*   `bench_files` (Dict): Dictionary of benchmark filenames.
*   `noise_path` (str): Path to noise spectra measurements.
*   `noise_number` (None | int): Specific session number for noise spectra.
*   `file_sig` (str): File signature for benchmark files.

**Methods:**
*   `__init__(self, snr_dict: Dict[int | str | float, Dict[int | str | float, List[str]]], bench_path: str = "", noise_path: str = "", noise_number: None | int = None, file_sig: str = "gasx")`: Initializes the `CEPAS_SNR_bench` object.
*   `set_noise_path(self, path: str) -> None`: Sets the path for noise spectra.
*   `set_bench_path(self, path: str) -> None`: Sets the path for benchmark spectra.
*   `make_bench(self, pressure: int | str | float, frequency: int | str | float) -> CEPAS_benchmark`: Creates a `CEPAS_benchmark` object for chemical spectra.
*   `make_noise_bench(self, pressure: int | str | float, frequency: int | str | float) -> CEPAS_benchmark`: Creates a `CEPAS_benchmark` object for single-point noise.
*   `get_signal(self, start: int | float, end: int | float, pressure: int | str | float, frequency: int | str | float, n_knots: int = 11, coly: str = "H2_pnorm") -> Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame, float]`: Returns the chosen signal peak in a spectral region.
*   `get_horizontal_noise(self, start: int | float, end: int | float, pressure: int | str | float, frequency: int | str | float, wstart: int | float = 1687.5, wend: int | float = 1707.5, n_knots: int = 10)`: Calculates horizontal noise in a specified window.
*   `get_vertical_noise(self, start: int | float, end: int | float, pressure: int | str | float, frequency: int | str | float, wstart: int | float = 1687.5, wend: int | float = 1707.5, n_knots: int = 10) -> Tuple[List[str] | List[float], float | np.floating[Any]]`: Calculates vertical noise in a specified window.
*   `get_single_point_noise(self, pressure: int | str | float, frequency: int | str | float) -> Tuple[pd.DataFrame, float, pd.DataFrame]`: Returns filtered single-point noise data and its standard deviation.
*   `get_clean_noise(self, pressure: int | str | float, frequency: int | str | float) -> Tuple[None, float]`: Returns the mean clean noise from noise spectra data.
*   `get_all_snrs(self, start: List[int | float], end: List[int | float], h: int = 2, n_knots: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]`: Creates DataFrames for SNR and noise, organized by pressure and modulation frequency.
*   `get_mag_snrs(self, start: int | float, end: int | float, n_knots: int = 10, h: int = 2, signal_col: str = "magnitude_pnorm", skip_single: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: Creates DataFrames for SNR, noise, and signal for magnitude spectra.

### `my_utils/common.py`

This module provides general utility functions.

#### `std_range` Function

Filters data based on the Interquartile Range (IQR) and calculates the standard deviation of the filtered data.

*   `std_range(df: pd.DataFrame, y: str, coef: int | float = 1) -> Tuple[pd.DataFrame, float, pd.DataFrame]`
    *   `df`: Input DataFrame.
    *   `y`: Column name of interest.
    *   `coef`: Coefficient for IQR filtering (larger makes filtering weaker).
    *   Returns: Filtered DataFrame, new standard deviation, and outliers.

#### `dir_match` Function

Iterates through a directory's contents and returns a list of filenames that match a given regex pattern.

*   `dir_match(path: str, pattern: Pattern[str]) -> List[str]`
    *   `path`: Path to the directory.
    *   `pattern`: Regex pattern to match.
    *   Returns: List of matching filenames.

### `my_utils/defs.py`

This module contains project-wide constants, plotting utilities, and specialized functions for spectroscopic analysis.

#### Constants

*   `G_MODEL_ID`, `G_API_KEY`, `G_CLIENT`: Google Gemini API setup parameters.
*   `col_names`: Dictionary mapping column indices to names for data loading.

#### Functions

*   `hello_test() -> None`: A simple test function that prints "Hello!".
*   `draw_points(coords: Dict[str, List[float | int]], x_mesh: np.ndarray, y_mesh: np.ndarray, bound_0: int | float, bound_1: int | float, n_points: int, ax: matplotlib.axes.Axes) -> None`: Draws 3D points and connecting lines on a given matplotlib axis.
*   `add_point(coords: Dict[str, List[float | int]], point: List[float | int], dims: int = 3) -> None`: Adds an N-dimensional point to a dictionary of coordinates.
*   `create_masks(grid: Tuple[np.ndarray, np.ndarray], verts: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]`: Creates two boolean masks for a grid: one for points inside a defined polygonal path and one for points outside.
*   `phase_correct(df: pd.DataFrame, y: str, x: str) -> Tuple[pd.Series, pd.Series, float]`: Performs phase correction on spectral data based on two intensity columns.
*   `std_range(df: pd.DataFrame, y: str, coef: int | float = 1) -> Tuple[pd.DataFrame, float, pd.DataFrame]`: (Duplicate of `common.py`'s `std_range`) Filters data based on IQR and calculates new standard deviation.
*   `dir_match_dict(path: str, patterns: Dict[int | str | float, Dict[int | str | float, Pattern[str]]]) -> Dict[int | str | float, Dict[int | str | float, List[str]]]`: Organizes filenames into a nested dictionary structure based on provided regex patterns.
*   `create_regex_strings(list1: List[int], list2: List[int], f: str = "gasx") -> Dict[int | str | float, Dict[int | str | float, Pattern[str]]]`: Generates a nested dictionary of regex patterns for file matching based on two lists (e.g., pressures and frequencies).
*   `gaussian_hwhm(v0, T, M) -> float`: Calculates the Gaussian (Doppler) Half-Width at Half-Maximum (HWHM) in cm^-1.
*   `lorentzian_hwhm(P_atm, T, gamma_L0, T0, n) -> float`: Calculates the Lorentzian (Pressure) HWHM in cm^-1.
*   `get_n_strongest(molecule_name: str, molar_mass: float, start: float, end: float, n: int) -> List[Dict[str, Any]]`: Fetches the `n` most intensive line parameters for a specified molecule from the HITRAN database within a given wavenumber range.
*   `dir_match(path: str, pattern: Pattern[str] | str) -> list[str]`: (Duplicate of `common.py`'s `dir_match`) Iterates through a directory's contents and returns a list of filenames that match a given regex pattern.
*   `subtract_hex(n1: str, n2: str) -> str`: Performs hexadecimal subtraction.
*   `sum_hex(n1: str, n2: str) -> str`: Performs hexadecimal addition.
*   `get_wavenumber(unit1: float, units1: List[float], units2: List[float]) -> List[float]`: Converts a value from `units1` to `units2` based on a linear regression fit.
*   `get_modamp_around_line_in_wav(modamps: List[str], line_in_offsets: float, conversion_params: Tuple[float, float]) -> List[float]`: Converts modulation amplitudes from LabView hexadecimal values to wavenumbers (cm^-1).
*   `amp_test_plot(pressure_string: str | int, path: str, amp_test_dict: Dict, used_amps: List[str] | List[int], vlinepos: float, ow_params: Tuple[List[float], List[float]], ctu) -> None`: Generates a plot to visualize amplitude test results for a specific spectral line.
*   `extract_PT_ratios(tests: dict) -> dict`: Extracts peak-to-trough ratios from `CEPAS_measurement` objects.
*   `askai(prompt: str) -> str | None`: A simple wrapper function to send a prompt to the Gemini AI model and return its text response.
*   `defai(prompt: str, defname: str) -> str | None`: Generates a Python function (as a string) with docstrings and type hints based on a given prompt, using the Gemini AI model.
*   `get_voigt(v0: float, sigma_g: float, gamma_l: float, wavenumber_axis: np.ndarray | float = np.linspace(6983.4, 6984, 1000)) -> np.ndarray`: Generates a Voigt profile given line center, Gaussian HWHM, and Lorentzian HWHM.
*   `get_range(df: pd.DataFrame, axis: str, ax0: float, ax1: float) -> pd.DataFrame`: Returns a subset of a DataFrame where a specified column's values fall within a given range.
*   `voigt_fit(f, xdata, ydata) -> Tuple[np.ndarray, np.ndarray]`: Performs a curve fit using the provided function `f` and data.
*   `get_fwhm_theory(xdata: np.ndarray, ydata: np.ndarray) -> float`: Calculates the Full Width at Half Maximum (FWHM) for spectral line-like data.
