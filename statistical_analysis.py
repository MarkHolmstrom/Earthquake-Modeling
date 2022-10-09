#!/usr/bin/env python3

import numpy as np


def b_value_maximum_likelihood(magnitudes):
    """Calculates the b-value given earthquake magnitudes using the maximum likelihood method described by
    bval_maxlkh2.m
    Expected input is a NumPy array."""
    neq = len(magnitudes)
    mean_magnitude = np.mean(magnitudes)
    min_magnitude = np.min(magnitudes)
    b_value = (1/(mean_magnitude-min_magnitude))*np.log10(np.exp(1))
    a_value = np.log10(neq) + b_value * min_magnitude
    std_dev = np.sum(np.square(magnitudes-mean_magnitude)) / (neq * (neq - 1))
    std_err = 2.30 * np.sqrt(std_dev) * b_value ** 2
    return b_value, a_value, std_err


def b_value_least_squares_regression(magnitudes, bin_size=0.2, precision=3):
    """Calculating the b-value given earthquake magnitudes using the least squares regression fit method described by
    bval_lsqreg
    Expected input is a NumPy array and, optionally, two scalars: the bin size and decimal place precision."""
    scaling_factor = 10 ** precision
    magnitudes = (scaling_factor * magnitudes).astype(int)
    bin_size = int(scaling_factor * bin_size)
    # Rescaling and integer conversion to avoid float precision problems
    max_magnitude = np.max(magnitudes)
    min_magnitude = np.min(magnitudes)
    sx, sy, sxy, sxx = 0, 0, 0, 0
    x, y = np.array([]), np.array([])
    count_range = np.arange(min_magnitude, max_magnitude + bin_size, bin_size)
    if count_range[-1] > max_magnitude:
        count_range = count_range[:-1]
    # Exactly match Matlab's count=mmin:bin:mmax
    for i in count_range:
        x = np.append(x, i)
        y_lin = np.sum(magnitudes >= i)  # bool count
        if y_lin > 0:
            y = np.append(y, np.log10(y_lin))
        else:
            y = np.append(y, 0)
        sx += x[-1]
        sy += y[-1]
        sxy += x[-1] * y[-1]
        sxx += x[-1] ** 2
    num_bins = len(x)
    numerator, denominator = 0, 0
    for i, j in enumerate(count_range):
        numerator += (x[i] - sx / num_bins) * (y[i] - sy / num_bins)
        denominator += (x[i] - sx / num_bins) ** 2
    b_value = -numerator / denominator
    a_value = sy / num_bins + sx / num_bins * b_value
    return b_value * scaling_factor,  a_value  # rescale back the b-value
