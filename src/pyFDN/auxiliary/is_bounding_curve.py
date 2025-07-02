import numpy as np
from scipy.interpolate import interp1d

def is_bounding_curve(x_points, y_points, x_curve, y_curve, bound_type):
    """
    Check if all value points are bounded by the curve.
    Args:
        x_points: x-coordinates of data points (1D array)
        y_points: y-coordinates of data points (1D array)
        x_curve: x-coordinates of curve points (1D array)
        y_curve: y-coordinates of curve points (1D array)
        bound_type: 'upper' or 'lower'
    Returns:
        all_bounded: bool, whether all data points are bounded
        is_bounded: boolean array, whether each data point is bounded
    """
    # Spline interpolation with extrapolation
    interp = interp1d(x_curve, y_curve, kind='cubic', fill_value='extrapolate')
    y_curve_interp = interp(x_points)

    if bound_type == 'upper':
        is_bounded = y_curve_interp >= y_points
    elif bound_type == 'lower':
        is_bounded = y_curve_interp <= y_points
    else:
        raise ValueError("bound_type must be 'upper' or 'lower'")

    all_bounded = np.all(is_bounded)
    return all_bounded, is_bounded
