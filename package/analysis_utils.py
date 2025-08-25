from scipy.signal import find_peaks, savgol_filter, peak_widths, peak_prominences
from pybaselines import Baseline
import numpy as np


def find_peaks_and_widths(
    twoth_values,
    intensity,
    smooth=True,
    width_at_height=0.8,
    wlen=None,
    find_peaks_dict={"prominence": 100},
):
    """
    Find peaks and their widths in the given intensity data.
    Parameters
    ----------
    twoth_values : array-like
        The 2θ values corresponding to the intensity data.
    intensity : array-like
        The intensity data in which to find peaks.
    width_at_height : float, optional
        The relative height at which to measure the peak width (default is 0.8).
    wlen : int, optional
        The length of the filter window for smoothing the intensity data in degrees (default is None, which uses a default value).
    find_peaks_dict : dict, optional
        Additional parameters to pass to scipy.signal.find_peaks (default is {"prominence": 100}).
    Returns
    -------
    tuple
        A tuple containing:
        - peak positions (2θ values and intensities)
        - peak widths in degrees
        - width heights
        - left intercepts (2θ values)
        - right intercepts (2θ values)
        - prominences
    """
    if smooth:
        smoothed_intensity = savgol_filter(intensity, window_length=11, polyorder=3)
    else:
        smoothed_intensity = intensity

    deg_per_index_step = (twoth_values[-1] - twoth_values[0]) / (len(twoth_values) - 1)
    if wlen is not None:
        wlen = int(wlen / deg_per_index_step)

    peaks, _ = find_peaks(smoothed_intensity, **find_peaks_dict)
    prominences = peak_prominences(smoothed_intensity, peaks, wlen=wlen)
    widths_result = peak_widths(
        smoothed_intensity,
        peaks,
        rel_height=width_at_height,
        prominence_data=prominences,
    )

    results = []
    for i in range(len(peaks)):
        peak_intensity = np.max(
            intensity[
                widths_result[2][i].astype(int) : widths_result[3][i].astype(int) + 1
            ]
        )
        properties = {
            "width": widths_result[0][i] * deg_per_index_step,
            "width_height": widths_result[1][i],
            "left_ip": twoth_values[widths_result[2][i].astype(int)],
            "right_ip": twoth_values[widths_result[3][i].astype(int)],
            "prominence": prominences[0][i],
        }
        results.append(((twoth_values[peaks[i]], peak_intensity), properties))

    return tuple(zip(*results))


def correct_baseline(
    twoth_values, intensity, offset=100, min_clip=None, method=None, **kwargs
):
    """
    Correct the baseline of the intensity data using the specified method.

    Parameters
    ----------
    twoth_values : array-like
        The 2θ values corresponding to the intensity data.
    intensity : array-like
        The intensity data to correct.
    method : str
        The method to use for baseline correction.
    **kwargs : dict
        Additional parameters for the baseline correction method.

    Returns
    -------
    array-like
        The corrected intensity data.
    """

    if method is None:
        method = "iarpls"
        kwargs = {"lam": 5e5}

    baseline_fitter = Baseline(
        x_data=twoth_values, check_finite=False, assume_sorted=True
    )
    func = getattr(baseline_fitter, method)
    calc_baseline, params = func(intensity, **kwargs)
    corrected_intensity = (
        np.clip(intensity - calc_baseline, a_min=min_clip, a_max=None) + offset
    )

    return corrected_intensity, (calc_baseline, params)
