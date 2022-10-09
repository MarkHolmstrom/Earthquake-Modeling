#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph.opengl as gl  # opengl needs to be installed (as pyopengl) manually
from PyQt5.QtWidgets import QApplication, QFileDialog

import process_input


def draw_utm_north_pointer(view, y_range, x_centre, y_centre):
    """Draws a red sphere on the "top" of the grids to represent North."""
    north_pointer_mesh = gl.MeshData.sphere(rows=8, cols=16, radius=5)
    north_pointer = gl.GLMeshItem(meshdata=north_pointer_mesh, shader="shaded", smooth=True, glOptions="opaque",
                                  drawFaces=True, drawEdges=False, color=(255, 0, 0, 1))
    view.addItem(north_pointer)
    north_pointer.translate(x_centre, y_centre + y_range / 2 + 5, 0)


def draw_utm_grids(view, x_range, y_range, z_range, x_centre, y_centre, z_centre, spacing=10):
    """Draws a grid across the xy plane centered around the middle of the data and depth/altitude 0.
    The grid spans the range of the data, and its gridlines are 10 units (km) apart by default."""
    xy_grid = gl.GLGridItem()
    view.addItem(xy_grid)
    xy_grid.setSize(x_range, y_range)  # large enough to cover all the data
    xy_grid.setSpacing(spacing, spacing)
    xy_grid.translate(x_centre, y_centre, 0)

    depth_grid1 = gl.GLGridItem()
    view.addItem(depth_grid1)
    depth_grid1.rotate(90, 1, 0, 0)  # transformation order matters
    depth_grid1.setSize(x_range, z_range)  # large enough to cover all the data
    depth_grid1.setSpacing(spacing, spacing)
    depth_grid1.translate(x_centre, y_centre, z_centre)

    depth_grid2 = gl.GLGridItem()
    view.addItem(depth_grid2)
    depth_grid2.rotate(90, 0, 0, 1)
    depth_grid2.rotate(90, 0, 1, 0)  # second rotation aligns to x-axis
    depth_grid2.setSize(y_range, z_range)  # large enough to cover all the data
    depth_grid2.setSpacing(spacing, spacing)
    depth_grid2.translate(x_centre, y_centre, z_centre)


def draw_utm(view, df):
    """Calculates summary statistics used for drawing UTM components and calls specific functions to do so."""
    x_range = np.ceil((df["X"].max() - df["X"].min()) / 20) * 20
    y_range = np.ceil((df["Y"].max() - df["Y"].min()) / 20) * 20
    z_range = np.ceil(0 - df["Z"].min() / 20) * 20
    x_centre = (df["X"].max() + df["X"].min()) / 2
    y_centre = (df["Y"].max() + df["Y"].min()) / 2
    z_centre = -z_range / 2
    # The ceiling is 20 units for aesthetics, 10 results in grid lines of perpendicular grids not matching
    draw_utm_grids(view, x_range, y_range, z_range, x_centre, y_centre, z_centre)
    draw_utm_north_pointer(view, y_range, x_centre, y_centre)


def draw_earth(view):
    """Draws a wireframe mesh of the Earth and gridlines to span it centered around the origin."""
    xgrid = gl.GLGridItem()
    ygrid = gl.GLGridItem()
    zgrid = gl.GLGridItem()

    view.addItem(xgrid)
    view.addItem(ygrid)
    view.addItem(zgrid)

    xgrid.rotate(90, 0, 1, 0)
    ygrid.rotate(90, 1, 0, 0)

    xgrid.setSize(12800, 12800)
    ygrid.setSize(12800, 12800)
    zgrid.setSize(12800, 12800)

    xgrid.setSpacing(200, 200)
    ygrid.setSpacing(200, 200)
    zgrid.setSpacing(200, 200)
    # Grid lines are 200 units (kilometres) apart

    sphere_mesh = gl.MeshData.sphere(rows=180, cols=360, radius=6371)
    view.addItem(gl.GLMeshItem(meshdata=sphere_mesh, shader=None, smooth=False, glOptions="translucent",
                               drawFaces=False, drawEdges=True, edgeColor=(1, 1, 1, 0.1)))
    # Wireframe sphere to represent Earth - not an ellipsoid so inconsistent with calculated positions


def draw_plot(df, mode="u"):
    """Draws a 3D scatter plot. The mode is the projection: Earth or UTM.
    Expected input is a Pandas DataFrame object that has "X", "Y", "Z", "Time", and "Magnitude" columns."""

    view = gl.GLViewWidget()
    view.show()

    centre = df["X"].mean(), df["Y"].mean(), df["Z"].mean()
    view.pan(*centre)  # unpacking values
    # Centering camera

    if mode == "e":
        draw_earth(view)
    elif mode == "u":
        draw_utm(view, df)

    df["Colour"] = (df["Time"] - df["Time"].min()) / (df["Time"].max() - df["Time"].min())  # normalizing to [0,1]
    scatter_plot_colours = plt.cm.plasma(df["Colour"])
    scatter_plot_colours[:, 3] = 0.5
    # Data points are scaled from 0 to 1 by relative time for colouring
    # Blue denotes closer to beginning, yellow denotes closer to end
    # Transparency reset to the default value

    data = gl.GLScatterPlotItem(pos=np.transpose([df["X"], df["Y"], df["Z"]]), pxMode=False,
                                size=np.squeeze(np.transpose([df["Magnitude"]])) ** 2 / 100,
                                color=scatter_plot_colours)
    # Enabling pxMode (and rescaling sizes to not have division by 100) results in faster rendering with less precision
    # Scatter plot point size chosen empirically, it may be better to normalize by df["Mw"].mean() in general
    # Squaring the size parameter highlights large magnitudes
    view.addItem(data)


def main():
    app = QApplication(sys.argv)
    file_name = QFileDialog().getOpenFileName(filter="CSV Files (*.csv);;All Files (*)")[0]  # technically a path object

    if file_name == "":  # if cancelled
        print("\nGoodbye!")
        sys.exit()

    mode = None
    while mode not in {"e", "u"}:
        mode = input("Enter e for projecting data relative to a spherical Earth model, "
                     "u for projecting data relative to a UTM coordinate model: ")
    df = process_input.process(file_name, mode)
    draw_plot(df, mode)

    print("\nGoodbye!")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
