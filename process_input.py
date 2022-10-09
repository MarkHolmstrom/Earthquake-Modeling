#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd

import geospatial_conversion


def magnitude_of_completeness(df):
    """Removes data below the magnitude of completeness."""
    moc = None
    while type(moc) is not float:
        try:
            moc = input("Enter the magnitude of completeness or leave the input blank for no filtering: ")
            moc = float(moc)
            df = df.drop(df.loc[df["Magnitude"] < moc].index)
        except ValueError:
            if moc == "":
                break
    return df


def remove_outliers_distance_matrix(df, remove=0.01):
    """Removes outliers using a distance matrix analysis.
    The parameter remove controls the fraction of data to be removed as outliers. Not feasible for large datasets."""
    from scipy.spatial import distance_matrix
    location_vectors = df.loc[:, ["X", "Y", "Z"]]
    d_matrix = pd.DataFrame(distance_matrix(location_vectors, location_vectors), columns=df.index, index=df.index)
    total_distances = d_matrix.assign(Total=d_matrix[:].sum())["Total"]
    # sum of distances from individual earthquakes to all earthquakes
    total_distance_threshold = total_distances.quantile(1-remove)
    df = df[df.merge(total_distances, left_index=True, right_index=True)["Total"] < total_distance_threshold]
    # select rows only in which the total distance is less than the threshold
    return df


def remove_outliers_cropping(df):
    """Removes outliers using manual cropping."""
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle("Spread of Data in Spatial Dimensions")

    ax1.hist(df.X, log=False, bins=100)
    ax1.set_title("Easting")
    ax1.set(xlabel="X (km)", ylabel="Number of Events")

    ax2.hist(df.Y, log=False, bins=100)
    ax2.set_title("Northing")
    ax2.set(xlabel="Y (km)", ylabel="Number of Events")

    ax3.hist(df.Z, log=False, bins=100)
    ax3.set_title("Depth")
    ax3.set(xlabel="Z (km)", ylabel="Number of Events")

    plt.tight_layout()  # padding so that all text fits on default window
    plt.draw()
    plt.show(block=False)
    plt.pause(0.01)

    x_min, x_max, y_min, y_max, z_min, z_max = None, None, None, None, None, None
    while type(x_min) is not float:
        try:
            x_min = input("Enter minimum X value or leave the input blank for no cropping: ")
            x_min = float(x_min)
            df = df.drop(df.loc[df["X"] < x_min].index)
        except ValueError:
            if x_min == "":
                break
    while type(x_max) is not float:
        try:
            x_max = input("Enter maximum X value or leave the input blank for no cropping: ")
            x_max = float(x_max)
            df = df.drop(df.loc[df["X"] > x_max].index)
        except ValueError:
            if x_max == "":
                break
    while type(y_min) is not float:
        try:
            y_min = input("Enter minimum Y value or leave the input blank for no cropping: ")
            y_min = float(y_min)
            df = df.drop(df.loc[df["Y"] < y_min].index)
        except ValueError:
            if y_min == "":
                break
    while type(y_max) is not float:
        try:
            y_max = input("Enter maximum Y value or leave the input blank for no cropping: ")
            y_max = float(y_max)
            df = df.drop(df.loc[df["Y"] > y_max].index)
        except ValueError:
            if y_max == "":
                break
    while type(z_min) is not float:
        try:
            z_min = input("Enter minimum Z value or leave the input blank for no cropping: ")
            z_min = float(z_min)
            df = df.drop(df.loc[df["Z"] < z_min].index)
        except ValueError:
            if z_min == "":
                break
    while type(z_max) is not float:
        try:
            z_max = input("Enter maximum Z value or leave the input blank for no cropping: ")
            z_max = float(z_max)
            df = df.drop(df.loc[df["Z"] > z_max].index)
        except ValueError:
            if z_max == "":
                break

    plt.close()
    return df


def process(file_name, mode="u"):
    """Converts data for statistical analysis as necessary. Expected file format is a csv file with the columns
    "Year", "Month", "Day", "Hour", "Minute", "Second", "latitude", "Longitude", "Depth", and "Magnitude".
    Depth is to be expressed in kilometres."""
    df = pd.read_csv(file_name)
    df = df.dropna()  # removing entries with no associated values
    df["Time"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute", "Second"]], infer_datetime_format=True)
    df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute", "Second"])  # dropping redundant columns
    if mode == "e":  # earth mode
        df["X"], df["Y"], df["Z"] = geospatial_conversion.get_cartesian(
            df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), df["Depth"].to_numpy())
    elif mode == "u":  # utm mode
        df["X"], df["Y"], df["Z"] = geospatial_conversion.get_utm(
            df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), df["Depth"].to_numpy())
    df = df.drop(columns=["Latitude", "Longitude", "Depth"])  # dropping redundant columns
    df = magnitude_of_completeness(df)
    df = remove_outliers_cropping(df)
    return df
