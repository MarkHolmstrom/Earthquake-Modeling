#!/usr/bin/env python3

import numpy as np
import utm


def get_cartesian(latitude, longitude, depth):
    """Converts spherical to Cartesian coordinates using WGS84 ellipsoid constants,
    adapted from https://www.mathworks.com/matlabcentral/fileexchange/7942-covert-lat-lon-alt-to-ecef-cartesian
    Expected inputs are NumPy arrays."""
    latitude = (latitude * np.pi) / 180  # converting to radians
    longitude = (longitude * np.pi) / 180  # converting to radians
    a = 6378.137
    e = 8.1819190842622e-2
    N = a / np.sqrt(1 - e**2 * np.sin(latitude)**2)
    K = (N - depth) * np.cos(latitude)
    x = K * np.cos(longitude)
    y = K * np.sin(longitude)
    z = ((1 - e**2) * N - depth) * np.sin(latitude)
    return x, y, z


def get_utm(latitude, longitude, depth):
    """Converts spherical to UTM coordinates. Expected inputs are NumPy arrays."""
    x, y = utm.from_latlon(latitude, longitude)[:2]
    # get first two values from UTM coordinates, easting and northing
    z = -depth  # z is negative of depth
    return x / 1000, y / 1000, z  # rescale x and y to be in kilometres
