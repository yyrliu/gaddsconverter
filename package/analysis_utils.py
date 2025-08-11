from scipy.signal import find_peaks, savgol_filter, peak_widths, peak_prominences

def find_peaks_and_widths(twoth_values, intensity, width_at_height=0.8, find_peaks_dict={"prominence": 100}):
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
    
    smoothed_intensity = savgol_filter(intensity, window_length=51, polyorder=3)
    
    peaks, _ = find_peaks(smoothed_intensity, **find_peaks_dict)
    prominences = peak_prominences(smoothed_intensity, peaks)
    widths_result = peak_widths(smoothed_intensity, peaks, rel_height=width_at_height, prominence_data=prominences)

    deg_per_index_step = (twoth_values[-1] - twoth_values[0]) / (len(twoth_values) - 1)

    # peak positions, widths, width_heights, left_ips, right_ips, prominences
    return ((twoth_values[peaks], intensity[peaks]),
            widths_result[0] * deg_per_index_step,
            widths_result[1],
            twoth_values[widths_result[2].astype(int)],
            twoth_values[widths_result[3].astype(int)],
            prominences)
