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

# scipy
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from scipy.constants import k as k_B, c, N_A

# ML
from sklearn.linear_model import LinearRegression

# GenAI
from google import genai
# from google.genai import types

# natural data databases
import hapi


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


# --------------------- SEPARATE DEFINITIONS -------------------------

def hello_test():
    print("Hello!")


def phase_correct(df: pd.DataFrame,
                  y: str,
                  x: str) -> Tuple[pd.Series, pd.Series, float]:
    """
    phase correct the spectrum based on two pi/2 (90 degrees) offset
    intensity data columns. Starts at some angle, then optimizes
    again with resolution

    Args:
        df (pd.DataFrame): used data
        y (str): Name of y data
        x (str): Name of x data

    Returns:
        (Tuple[pd.Series, pd.Series, float, int]): Returns \
            phase corrected data: one should be true \
            spectrum, the other should be some small residual \
                near 0.
    """
    solution: Tuple[pd.Series, pd.Series, float] | None = None
    min_residual: float = np.inf
    for another_angle in np.linspace(0, 2*np.pi, 1000):

        correction_angle = np.atan2(df[y].mean(),
                                    df[x].mean()) - another_angle

        X_cor = np.cos(correction_angle)*df[y] + \
            np.sin(correction_angle)*df[x]
        Y_cor = -np.sin(correction_angle)*df[y] + \
            np.cos(correction_angle)*df[x]
        current_residual = abs(Y_cor.mean())

        if current_residual < min_residual:
            min_residual = abs(Y_cor.mean())
            solution = (X_cor, Y_cor, correction_angle)

    if solution is not None:
        abs_max = abs(solution[0].max())
        abs_min = abs(solution[0].min())

        if abs_min > abs_max:
            solution = (-solution[0], -solution[1], solution[2])

        return solution
    else:
        raise TypeError("Make sure that solution to phase correct is not None")


def std_range(df: pd.DataFrame,
              y: str,
              coef: int | float = 1) -> \
                Tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Filters data. Creates new stddev for some filtered data based on the IQR
    from 0.25 quantile to 0.75 quantile.

    Args:
        df (pd.DataFrame): pandas dataframe to work on
        y (str): column name of the column of interest
        coef (int | float): larger number makes filtering weaker (default=1)

    Returns:
        (Tuple[pd.DataFrame, float, pd.DataFrame]): new data frame, new stddev,
        outliers
    """
    Q1 = df[y].quantile(0.25)
    Q3 = df[y].quantile(0.75)
    IQR = Q3 - Q1
    new_df = df[(df[y] > Q1-coef*IQR) & (df[y] < Q3+coef*IQR)]
    outliers = df[(df[y] < Q1-coef*IQR) & (df[y] > Q3+coef*IQR)]
    new_std = new_df[y].std()
    return new_df, new_std, outliers


def dir_match_dict(path: str,
                   patterns: Dict[int | str | float,
                                  Dict[int | str | float,
                                       Pattern[str]]]) -> \
                                        Dict[int | str | float,
                                             Dict[int | str | float,
                                                  List[str]]]:
    """
    Gets out structured dictionary of dictionaries from
    `create_regex_strings()` return value.

    Args:
        patterns (Dict[int, Dict[int, List[Pattern[str]]]]): Output of
        `create_regex_strings`

    Returns:
        (Dict[int, Dict[int, List[str]]]): Lists of filenames in a structured
        manner

    ```
    {pressure:
        {frequency:
            [filename1, filename2, ...],
            .
            .
            .
        frequency_n:
            [...]},
    .
    .
    .
    pressure_n: ...}
    ```
    """
    filenames = {}
    for k1, v1 in patterns.items():
        filenames[k1] = {}
        for k2, v2 in v1.items():
            filenames[k1][k2] = [f for f in dir_match(path, v2)]

    return filenames


def create_regex_strings(list1: List[int],
                         list2: List[int],
                         f: str = "gasx") -> \
                            Dict[int | str | float,
                                 Dict[int | str | float,
                                      Pattern[str]]]:
    """
    Runs a nested for loop to mach each `list1` item with each
    `list2` item in an appropriate raw string to use with regex

    Args:
        list1 (List[int]): List, e.g. of pressures
        list2 (List[int]): List, e.g. of frequencies
        f (str): name flag, default "gasx"

    Returns:
        List[str]: The generated list of strings, to use wherever
    """
    patterns: Dict[int | str | float,
                   Dict[int | str | float,
                        Pattern[str]]] = {}
    for l1 in list1:
        patterns[l1] = {}
        for l2 in list2:
            current_string = re.compile(
                f"{f}_{l1}_(12|24|32)_{l2}__msr__[0-9]{{1,2}}"
                )
            patterns[l1][l2] = current_string
    return patterns


def gaussian_hwhm(v0, T, M):
    """Calculates the Gaussian (Doppler) HWHM in cm^-1."""
    # mass_of_molecule is the mass of a single molecule in kg
    mass_of_molecule = M / N_A
    return v0 * np.sqrt(2 * k_B * T * np.log(2) / (mass_of_molecule * c**2))


def lorentzian_hwhm(P_atm, T, gamma_L0, T0, n):
    """Calculates the Lorentzian (Pressure) HWHM in cm^-1."""
    # Pressure broadening coeff scaled by temperature
    gamma_L = gamma_L0 * (T0 / T)**n
    return P_atm * gamma_L


def get_n_strongest(molecule_name: str,
                    molar_mass: float,
                    start: float,
                    end: float, n: int) -> List[Dict[str, Any]]:
    """
    Gets `n` most intensive line parameters for molecule of `molecule_name`
    from HITRAN database in range
    from `start` to `end`, using `hapi` library

    Args:
        molecule_name (str): name of the molecule, e.g., `'h2o'`
        molar_mass (float): molar mass of the molecule in kg/mol
        start (float): start of the range to work on (in wavenumbers)
        end (float): end of the range to work on (in wavenumbers)
        n (int): amount of lines to return, starting with the most
        intensive lines

    Returns:
        List[Dict[str, Any]]: Returns a list of parameter dicts,
        w/ names as per `hapi` nomenclature
    """
    line_dicts = []

    MOLECULE_NAME = str.upper(molecule_name)
    hapi.fetch(MOLECULE_NAME, 1, 1, start, end)

    line_strengths = hapi.getColumn(MOLECULE_NAME, 'sw')  # sw = line intensity
    line_strengths_df = pd.DataFrame({
        "og_order": list(range(len(line_strengths))),
        "sw": line_strengths
    })
    strongest_line_indexes = line_strengths_df.sort_values(
        by="sw",
        ascending=False
        )[:n]['og_order']

    for strongest_line_index in strongest_line_indexes:

        MOL_LINE_PARAMS = {
            'v0': hapi.getColumn(
                MOLECULE_NAME, 'nu'
                )[strongest_line_index],  # Line center in cm^-1
            'gamma_L0': hapi.getColumn(
                MOLECULE_NAME, 'gamma_air'
                )[strongest_line_index],  # Air-broadening coeff at T0, P0
            'n_exp': hapi.getColumn(
                MOLECULE_NAME, 'n_air'
                )[strongest_line_index],  # Temperature exponent
            'T0': 296.0,  # Reference Temperature
            'P0': 1.0,  # Reference Pressure (atm)
            'M': molar_mass  # Molar mass of H2O in kg/mol
        }
        print(
            f"--- Fetched {molecule_name.upper(
            )} Line Parameters from HITRAN ---")
        for key, val in MOL_LINE_PARAMS.items():
            print(f"{key}: {val}")

        line_dicts.append(MOL_LINE_PARAMS)

    return line_dicts


def dir_match(path: str, pattern: Pattern[str] | str) -> list[str]:
    """
    Iterates through contents of dir and matches according
    to some regex pattern

    Args:
        path (str): path to directory of interest
        pattern (Pattern[str] | str): pattern to find, either simple or regex

    Returns:
        list[str]: All matches
    """
    matched = []
    dir_contents = sorted(os.listdir(path))
    for item in dir_contents:
        if re.match(pattern, item) is not None:
            matched.append(item)
    return matched


def subtract_hex(n1: str, n2: str) -> str:
    """
    Does subtraction in hex

    Args:
        n1 (str): First number as a string
        n2 (str): Second number as a string

    Returns:
        str: a hex subtraction
    """
    h1 = int(n1, 16)
    h2 = int(n2, 16)
    return hex(abs(h1-h2))


def sum_hex(n1: str, n2: str) -> str:
    """
    Sums two numbers in hex

    Args:
        n1 (str): First number as a string
        n2 (str): Second number as a string

    Returns:
        str: a hex subtraction
    """
    h1 = int(n1, 16)
    h2 = int(n2, 16)
    return hex(h1+h2)


def get_wavenumber(unit1: float,
                   units1: List[float],
                   units2: List[float]) -> List[float]:
    """
    Converts `unit1` to `unit2` based on the
    `sklearn.linear_model.LinearRegression()`
    fit of lists `units1` and `units2`

    Args:
        unit1 (float): Value of interest (unit from which will convert)
        units1 (list[float]): List of values in units of interest
        units2 (list[float]): List of values in units from which
        we are converting

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


def get_modamp_around_line_in_wav(modamps: List[str],
                                  line_in_offsets: float,
                                  conversion_params: Tuple[float, float]) \
                                    -> List[float]:
    """
    Effectively converts modulation amplitude
    from LabView hexes to [somewhat] actual wavenumbers cm-1

    Args:
        modamps (List[str]): Gives a list of modamps as hex
        strings (w/0 '0x' prepended)
        line_in_offsets (float): Line position in offsets (decimal)

    Returns:
        List[float]: Modulation amplitudes in wavenumbers cm-1

    """
    amp_wav_list = []
    for amp in modamps:
        offset = int(f"0x{amp}", 16)
        left_bound = line_in_offsets - offset/2
        right_bound = line_in_offsets + offset/2
        left_bound = conversion_params[0]*left_bound + \
            conversion_params[1]  # type: ignore
        right_bound = conversion_params[0]*right_bound + \
            conversion_params[1]  # type: ignore
        mod_amp = left_bound - right_bound
        amp_wav_list.append(mod_amp)
    return amp_wav_list


def amp_test_plot(pressure_string: str | int,
                  path: str,
                  amp_test_dict: Dict,
                  used_amps: List[str] | List[int],
                  vlinepos: float,
                  ow_params: Tuple[List[float], List[float]], ctu) -> None:
    """
    Do some parameter tests on specific line,
    For example, plot a line with different laser
    power/current modulation amplitudes or modulation frequencies

    Args:
        pressure_string (str | int): used to retrieve data according to
        pressure in filename
        path (str): used for underlying `CEPAS_measurement()` call
        ampt_test_dict (Dict): Just an empty dict, in which all used
        measurements are supposed to be stored (created globally before call)
        ow_params (Tuple[List[float], List[float]]): pass two lists in the
        tuple that will be used for a fit to add wavnumbers axis
        vlinepos (float): argument which can be useful with
        `ipywidgets.interact`-ish calls to find some positions

    Returns:
        None: Just plots multiple plots on top of each other,
        with legend of different parameters used
    """
    maxs = []  # to draw
    mins = []  # a vertical red line
    plt.clf()
    plt.figure(figsize=(10, 10))
    for i in used_amps:
        # auto_freq_test[f'cm{i}_fl'] = CEPAS_measurement(
        # path=path2, path_signature=f"gasx_600_51_{i}__msr__", cols=col_names)
        amp_test_dict[f'cm{i}_fl'] = ctu(
            path=path,
            path_signature=f"gasx_{pressure_string}_{i}_20__msr__",
            cols=col_names)  # type: ignore
        amp_test_dict[f'cm{i}_fl'].spectra_list[0]['wav'] = \
            get_wavenumber(
                amp_test_dict[f'cm{i}_fl'].spectra_list[0]['offset1'],
                ow_params[0], ow_params[1])[0]  # type: ignore
        # auto_test[f'cm{i}_fl'].water_plot()
        # plt.plot(
        # auto_freq_test[f'cm{i}_fl'].spectra_list[0]['time'],
        # auto_freq_test[f'cm{i}_fl'].spectra_list[0]['RH'],
        # label=f"{i}")
        # plt.plot(
        # auto_freq_test[f'cm{i}_fl'].spectra_list[1]['time'],
        # auto_freq_test[f'cm{i}_fl'].spectra_list[1]['RH'],
        # label=f"{i}")
        # plt.plot(
        # auto_freq_test[f'cm{i}_fl'].spectra_list[2]['time'],
        # auto_freq_test[f'cm{i}_fl'].spectra_list[2]['RH'],
        # label=f"{i}")
        min_val = abs(
            amp_test_dict[f'cm{i}_fl'].spectra_list[0]['H2_pnorm'].min())
        max_val = abs(
            amp_test_dict[f'cm{i}_fl'].spectra_list[0]['H2_pnorm'].max())
        mins.append(min_val)
        maxs.append(max_val)
        ratio = max_val / min_val
        plt.plot(
            amp_test_dict[f'cm{i}_fl'].spectra_list[0]['wav'],
            amp_test_dict[f'cm{i}_fl'].spectra_list[0]['H2_pnorm'],
            label=f'Amplitude = {i}, PTR={ratio:.2f}')
    plt.legend(loc=2)
    plt.vlines(
        vlinepos,
        ymin=-np.max(np.array(mins)),
        ymax=np.max(np.array(maxs)),
        colors=['red']
        )
    plt.title(f"{pressure_string} mbar")
    plt.show()


def extract_PT_ratios(tests: dict) -> dict:
    """
    In the data of a single line, returns the ratio abs(max)/abs(min)

    Args:
        tests (dict): takes dict of the `CEPAS_measurement` objects

    Returns:
        dict: dictionary of each individual spectra PT ratio

    """
    for_barplot = {}
    for test in list(tests.keys()):
        print(test)
        for_barplot[test] = {}
        for subtest in tests[test]:
            min_val = abs(
                tests[test][subtest].spectra_list[0]['H2_pnorm'].min())
            max_val = abs(
                tests[test][subtest].spectra_list[0]['H2_pnorm'].max())
            ratio = max_val / min_val
            key = f"Amplitude: {subtest}"
            value = ratio
            for_barplot[test][key] = value

    return for_barplot


def askai(prompt: str) -> str | None:
    """
    Simple wrapper to ask questions to gemini with just a string
    as argument
    """
    model = G_MODEL_ID
    client = G_CLIENT

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text


def defai(prompt: str, defname: str) -> str | None:
    """
    Returns a string that you can directly pass to `exec()`,
    provided you have defined `G_MODEL_ID`, `G_CLIENT` and have an
    api key from Google gemini
    """
    model = G_MODEL_ID
    client = G_CLIENT
    response = client.models.generate_content(
        model=model,
        contents=f"\
            Provide docstringed and type-hinted python function named \
                '{defname}' for the following prompt (in the output \
                    include only the code text!!!):\n" + prompt
    )
    to_exec: str | None = response.text
    if to_exec.startswith(  # type: ignore
            "```python"
            ) and to_exec.endswith("```"):  # type: ignore
        to_exec = to_exec[len("```python"):-len("```")].strip()  # type: ignore
        print("Stripped!")
    return to_exec  # type: ignore


def get_voigt(
        v0: float,
        sigma_g: float,
        gamma_l: float,
        wavenumber_axis:
        np.ndarray | float = np.linspace(6983.4, 6984, 1000)) -> np.ndarray:
    relative_wavenumber = wavenumber_axis - v0
    return voigt_profile(relative_wavenumber, sigma_g, gamma_l)


def get_range(df: pd.DataFrame,
              axis: str,
              ax0: float,
              ax1: float) -> pd.DataFrame:
    """
    Returns some slice of dataframe according to
    the slice of some column

    Args:
        df (pd.DataFrame): Pandas dataframe to work on
        axis (str): The name of the criterion column
        ax0 (float): Low bound of criterion column
        ax1 (float): High bound of criterion column

    Returns:
        (pd.DataFrame): the frame according to the slice of criterion \
        column
    """
    ddf = df[df[axis] > ax0 and df[axis] < ax1]
    return ddf


def voigt_fit(f, xdata, ydata) -> Tuple[np.ndarray, np.ndarray]:
    popt, pcov = curve_fit(f, xdata, ydata)
    return popt, pcov


def get_fwhm_theory(xdata: np.ndarray, ydata: np.ndarray) -> float:
    """
    Returns a value of full width at half maximum of spectral line-like data

    Args:
        xdata (float): x data
        ydata (float): y data

    Returns:
        List: [voigt fit, fwhm]

    """
    min_diff_x0 = 1
    min_diff_x1 = 1
    found_x0 = 0
    found_x1 = 0
    for idx in range(len(ydata)):
        diff = ydata.max()/2 - ydata[idx]
        if idx <= np.where(ydata == ydata.max())[0][0]:
            if abs(diff) < min_diff_x0:
                min_diff_x0 = abs(diff)
                found_x0 = xdata[idx]
            else:
                continue
        else:
            if abs(diff) < min_diff_x1:
                min_diff_x1 = abs(diff)
                found_x1 = xdata[idx]
    return found_x1-found_x0
