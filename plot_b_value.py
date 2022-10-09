#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def graph_xy(df, all_data, window_interval):
    """Displays a bar graph of b-values against their respective xy windows. Expected input is a Pandas DataFrame
        with columns "Window_x", "Window_y", "B_lsr", "B_ml", "Event_count", denoting middle x and y coordinates,
        b-value of least squares regression, b-value of maximum likelihood, and events per window respectively.
        Additionally, a Pandas dataframe containing the entire dataset with columns "X" and "Y" and a scalar denoting
        window interval, assuming square windows.
    """
    fig, axes = plt.subplots(ncols=4, nrows=2, gridspec_kw={"width_ratios": [15, 1, 15, 1]})  # columns for colour bars
    fig.suptitle("B-values of Windows")
    axes[0, 0].set_title("B-value from Least Squares Regression")
    axes[0, 2].set_title("B-value from Maximum Likelihood")
    axes[1, 0].set_title("Event Count")
    axes[1, 2].set_title("All Events")
    axes[1, 3].remove()  # redundant axe since the scatter plot doesn't have a colour bar
    df = df.astype({"Window_x": int, "Window_y": int})  # converting to integers for neater display
    df_lsr = df[["Window_x", "Window_y", "B_lsr"]].reset_index().pivot(
        index="Window_y", columns="Window_x", values="B_lsr").sort_index(ascending=False)
    # converting to wide-spread form for LSR visualisation
    # sorting by descending order to make heatmap coordinates match geographical coordinates
    df_ml = df[["Window_x", "Window_y", "B_ml"]].reset_index().pivot(
        index="Window_y", columns="Window_x", values="B_ml").sort_index(ascending=False)
    df_count = df[["Window_x", "Window_y", "Event_count"]].reset_index().pivot(
        index="Window_y", columns="Window_x", values="Event_count").sort_index(ascending=False)

    sns.heatmap(df_lsr, cmap="viridis", square=True, ax=axes[0, 0], cbar_ax=axes[0, 1])
    sns.heatmap(df_ml, cmap="viridis", square=True, ax=axes[0, 2], cbar_ax=axes[0, 3])
    sns.heatmap(df_count, cmap="viridis", square=True, ax=axes[1, 0], cbar_ax=axes[1, 1])
    # default behaviour of not colouring in null values is good
    axes[1, 2].scatter(all_data["X"], all_data["Y"], s=1)  # scatter plot of all data points
    axes[1, 2].set_xlim(df["Window_x"].min() - 0.5 * window_interval, df["Window_x"].max() + 0.5 * window_interval)
    axes[1, 2].set_ylim(df["Window_y"].min() - 0.5 * window_interval, df["Window_y"].max() + 0.5 * window_interval)
    # cannot just copy lims of other axes, heatmaps are discrete and use window numbers as indices internally
    # more accurate to window heatmaps than just taking min and max of all_data
    axes[1, 2].set_aspect(axes[0, 0].get_aspect())
    # match display of other three axes
    axes[0, 0].set(xlabel="X (km)", ylabel="Y (km)")
    axes[0, 2].set(xlabel="X (km)", ylabel="Y (km)")
    axes[1, 0].set(xlabel="X (km)", ylabel="Y (km)")
    axes[1, 2].set(xlabel="X (km)", ylabel="Y (km)")
    plt.tight_layout()  # padding so that all text fits on default window, mostly
    plt.show()


def scatter_plot_time(df):
    """Displays a scatter plot of b-values against their respective time windows. Expected input is a Pandas
    DataFrame with columns "Window", "B_lsr", "B_ml", "Min_time", "Max_time", denoting average event time,
    b-value of least squares regression, b-value of maximum likelihood, minimum and maximum event times of the window
    respectively. """
    plt.title("B-values Across Time Windows")
    plt.xlabel("Time Window")
    plt.ylabel("B-value")
    plt.gcf().autofmt_xdate()  # date-like formatting

    error_bar = [df.Window - df.Min_time,
                 df.Max_time - df.Window]
    plt.errorbar(df.Window, df.B_lsr, xerr=error_bar, fmt="o", c="red", label="Least Squares Regression")
    plt.errorbar(df.Window, df.B_ml, xerr=error_bar, fmt="o", c="blue", label="Maximum Likelihood")

    plt.legend()
    plt.tight_layout()  # padding so that all text fits on default window
    plt.show()


def line_plot_regression(df, b_value_lsr, a_value_lsr, b_value_mlk, a_value_mlk):
    """Displays the line plot for b-value calculations. Expected input is a Pandas DataFrame with a "Magnitude"
    column and appropriately named scalars"""
    df = df.sort_values(by="Magnitude", ascending=False, ignore_index=True)

    plt.xlabel("Magnitude")
    plt.ylabel("LogN")
    plt.plot(df.Magnitude, np.log10(df.index.to_numpy() + 1), label="Raw Data")
    # adding 1 to index avoids division by 0
    plt.legend()

    x = df.Magnitude.min(), df.Magnitude.max()
    y = a_value_lsr - b_value_lsr * df.Magnitude.min(), a_value_lsr - b_value_lsr * df.Magnitude.max()
    plt.plot(x, y, label="Least Squares Regression")

    y = a_value_mlk - b_value_mlk * df.Magnitude.min(), a_value_mlk - b_value_mlk * df.Magnitude.max()
    plt.plot(x, y, label="Maximum Likelihood")

    plt.legend(loc="upper right")
    plt.show()
