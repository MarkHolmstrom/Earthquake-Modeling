#!/usr/bin/env python3

import sys
import itertools

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog

import plot_b_value
import process_input
import statistical_analysis


def spatial_window_export(exported_data):
    """Exports data from spatial moving window analysis."""
    file_name = QFileDialog.getSaveFileName(filter="CSV Files (*.csv);;All Files (*)")[0]  # technically a path object
    if file_name != "":
        df_export = pd.DataFrame(exported_data,
                                 columns=["X_min", "X_max", "Y_min", "Y_max", "Z_min", "Z_max", "Event_count",
                                          "B_lsr", "A_lsr", "B_ml", "A_ml", "Std_err_ml"])
        df_export.to_csv(file_name, encoding="utf-8", index=False)


def spatial_window_menu():
    """Interactive handler for spatial moving window settings."""
    window_size = None
    while type(window_size) is not float:
        try:
            window_size = float(input("Enter the size of the (square) windows in km: "))
        except ValueError:
            pass
    window_step = None
    while type(window_step) is not float:
        try:
            window_step = float(input("Enter the increment of consecutive windows in km: "))
        except ValueError:
            pass
    window_min_count = None
    window_min_count_default = 500
    while type(window_min_count) is not int:
        try:
            window_min_count = input("Enter the minimum number of events per window or leave the input blank "
                                     "for the default " + str(window_min_count_default) + ": ")
            window_min_count = int(window_min_count)
        except ValueError:
            if window_min_count == "":
                window_min_count = window_min_count_default
    return window_step, window_size, window_min_count


def spatial_window(df, bin_size):
    """Analyses b-values of spacial windows."""
    window_step, window_size, window_min_count = spatial_window_menu()
    plotted_data = []
    exported_data = []
    x_array = np.arange(df["X"].min(), df["X"].max(), window_step)
    y_array = np.arange(df["Y"].min(), df["Y"].max(), window_step)
    coordinates = np.array(list(itertools.product(x_array, y_array)))  # all bottom right window anchor positions
    windows = [df[(df["X"].between(x, x + window_size)) &  # & instead of and is required
                  (df["Y"].between(y, y + window_size))] for x, y in coordinates]
    for i, window in enumerate(windows):
        b_value_lsr, b_value_ml, event_count = None, None, None
        if len(window) >= window_min_count:
            event_count = len(window)  # some redundancy for proper heatmap display
            # print("From", str(window["X"].min()), "to", str(window["X"].max()),
            #      "in X and", str(window["Y"].min()), "to", str(window["Y"].max()), " in Y the b-value is: ")
            b_value_lsr, a_value_lsr = statistical_analysis.b_value_least_squares_regression(window["Magnitude"]
                                                                                             .to_numpy(), bin_size)
            b_value_ml, a_value_ml, std_err_ml = statistical_analysis.b_value_maximum_likelihood(
                window["Magnitude"].to_numpy())
            exported_data.append((window["X"].min(), window["X"].max(), window["Y"].min(), window["Y"].max(),
                                  window["Z"].min(), window["Z"].max(), event_count,
                                  b_value_lsr, a_value_lsr, b_value_ml, a_value_ml, std_err_ml))
            # print(b_value_lsr, "from least squares regression and", b_value_ml, "from maximum likelihood")
        plotted_data.append((coordinates[i, 0] + 0.5 * window_size, coordinates[i, 1] + 0.5 * window_size,
                             b_value_lsr, b_value_ml, event_count))
    if not exported_data:  # the previous if statement was never executed, i.e. len(window) < window_min_count for all
        print("No window has the minimum number of events required, try again with different window settings")
        return
    df_plot = pd.DataFrame(plotted_data, columns=["Window_x", "Window_y", "B_lsr", "B_ml", "Event_count"])
    plot_b_value.graph_xy(df_plot, df, window_step)
    spatial_window_export(exported_data)


def temporal_window_export(exported_data):
    """Exports data from temporal moving window analysis."""
    file_name = QFileDialog.getSaveFileName(filter="CSV Files (*.csv);;All Files (*)")[0]  # technically a path object
    if file_name != "":
        df_export = pd.DataFrame(exported_data, columns=["Time_min", "Time_max", "Event_count",
                                                         "B_lsr", "A_lsr", "B_ml", "A_ml", "Std_err_ml"])
        df_export.to_csv(file_name + ".csv", encoding="utf-8", index=False)


def temporal_window_menu(df):
    """Interactive handler for temporal moving window settings."""
    df = df.sort_values(by="Time")

    mode_window = None
    windows = []

    while mode_window not in {"e", "s"}:
        mode_window = input("Enter e for evenly split windows, s for sliding windows: ")

    if mode_window == "e":
        window_number = None
        while type(window_number) is not int:
            try:
                window_number = int(input("Enter the number of windows: "))
            except ValueError:
                pass
        windows = np.array_split(df, window_number)

    elif mode_window == "s":
        window_size = None
        while type(window_size) is not int:
            try:
                window_size = int(input("Enter the size of the windows: "))
            except ValueError:
                pass
        window_step = None
        while type(window_step) is not int:
            try:
                window_step = int(input("Enter the increment of consecutive windows: "))
            except ValueError:
                pass
        window_counter = 0
        while window_counter + window_size <= len(df):  # append windows while there are still remaining ones
            windows.append(df.iloc[window_counter:window_counter + window_size])
            window_counter += window_step
    return windows


def temporal_window(df, bin_size):
    """Analyses b-values of temporal windows."""
    windows = temporal_window_menu(df)
    plotted_data = []
    exported_data = []
    for window in windows:
        # print("From", str(window.iloc[0]["Time"]), "to", str(window.iloc[-1]["Time"]), "the b-value is: ")
        min_time, max_time = window.iloc[0]["Time"], window.iloc[-1]["Time"]
        b_value_lsr, a_value_lsr = statistical_analysis.b_value_least_squares_regression(window["Magnitude"]
                                                                                         .to_numpy(), bin_size)

        b_value_ml, a_value_ml, std_err_ml = statistical_analysis.b_value_maximum_likelihood(window["Magnitude"]
                                                                                             .to_numpy())

        plotted_data.append((window["Time"].mean(), b_value_lsr, b_value_ml, min_time, max_time))
        exported_data.append((min_time, max_time, len(window),
                              b_value_lsr, a_value_lsr, b_value_ml, a_value_ml, std_err_ml))

        # print(b_value_lsr, "from least squares regression and", b_value_ml, "from maximum likelihood")

    df_plot = pd.DataFrame(plotted_data, columns=["Window", "B_lsr", "B_ml", "Min_time", "Max_time"])
    plot_b_value.scatter_plot_time(df_plot)
    temporal_window_export(exported_data)


def menu(df):
    """Interactive handler for moving window settings."""
    bin_size = None
    default_bin_size = 0.2
    while type(bin_size) is not float:
        try:
            bin_size = input("Enter the bin size for least squares regression or leave the input blank for the default "
                             + str(default_bin_size) + ": ")
            bin_size = float(bin_size)
        except ValueError:
            if bin_size == "":
                bin_size = default_bin_size
    while True:
        try:
            i = None
            while i not in {"s", "t"}:
                i = input("Enter t for temporal variability, s for spatial variability along the xy-plane: ")
            if i == "t":
                while True:
                    try:
                        temporal_window(df, bin_size)
                    except KeyboardInterrupt:
                        print("\n")
                        break
            elif i == "s":
                while True:
                    try:
                        spatial_window(df, bin_size)
                    except KeyboardInterrupt:
                        print("\n")
                        break
        except KeyboardInterrupt:
            print("\n")
            break


def plot_regression(df):
    """Plots the least squares regression and maximum likelihood of b and a values for the entire data set"""
    b_value_lsr, a_value_lsr = statistical_analysis.b_value_least_squares_regression(df["Magnitude"].to_numpy())
    b_value_ml, a_value_ml, std_err_ml = statistical_analysis.b_value_maximum_likelihood(df["Magnitude"].to_numpy())
    plot_b_value.line_plot_regression(df, b_value_lsr, a_value_lsr, b_value_ml, a_value_ml)


def main():
    app = QApplication(sys.argv)  # just to keep QApplication in memory, a gui event loop with exec_() isn't needed
    print("Enter CTRL+C to go back to a previous menu at any input prompt")
    while True:
        try:
            file_name = QFileDialog().getOpenFileName(filter="CSV Files (*.csv);;All Files (*)")[0]
            # technically a path object
            if file_name == "":
                raise KeyboardInterrupt  # if cancelled, same as entering ctrl+c
            while True:
                try:
                    df = process_input.process(file_name, "u")  # always use UTM coordinates for analysis
                    plot_regression(df)
                    while True:
                        try:
                            menu(df)
                        except KeyboardInterrupt:
                            print("\n")  # printing a newline to make it clear the process flow was interrupted
                            break
                except KeyboardInterrupt:
                    print("\n")
                    break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit()
    # a function wrapper could be used instead of while loops with try and except blocks everywhere


if __name__ == "__main__":
    main()
