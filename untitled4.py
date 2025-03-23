#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:46:58 2025

@author: juanramonjimenezmogollon
"""

import numpy as np
import matplotlib.pyplot as plt

def enlarge_polygon(points, enlargement_factor):
    """
    Enlarges a polygon defined by a path of points.

    Args:
        points: A list or NumPy array of (x, y) coordinates defining the polygon.
        enlargement_factor: The factor by which to enlarge the polygon (e.g., 1.1 for 10% larger).

    Returns:
        A NumPy array of (x, y) coordinates representing the enlarged polygon.
    """
    points = np.array(points)
    center = np.mean(points, axis=0)  # Calculate the centroid

    enlarged_points = []
    for point in points:
        vector = point - center
        enlarged_point = center + vector * enlargement_factor
        enlarged_points.append(enlarged_point)

    return np.array(enlarged_points)

def plot_polygon(points, label, color):
    """Plots a polygon."""
    points = np.array(points)
    plt.plot(np.append(points[:, 0], points[0, 0]),
             np.append(points[:, 1], points[0, 1]),
             label=label, color=color)

# Example usage 1: Simple square
points_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
enlargement_factor_square = 1.2
enlarged_square = enlarge_polygon(points_square, enlargement_factor_square)

plot_polygon(points_square, "Original Square", "blue")
plot_polygon(enlarged_square, "Enlarged Square", "red")
plt.axis('equal')
plt.legend()
plt.title("Polygon Enlargement - Square")
plt.show()

# Example usage 2: More complex shape
points_complex = [(0, 0), (0.5, 0.2), (1, 0), (1.2, 0.5), (1, 1), (0.5, 0.8), (0, 1), (-0.2, 0.5)]
enlargement_factor_complex = 1.15
enlarged_complex = enlarge_polygon(points_complex, enlargement_factor_complex)

plot_polygon(points_complex, "Original Complex", "green")
plot_polygon(enlarged_complex, "Enlarged Complex", "orange")
plt.axis('equal')
plt.legend()
plt.title("Polygon Enlargement - Complex Shape")
plt.show()

# Example usage 3: Shrinking the polygon
points_complex2 = [(0, 0), (0.5, 0.2), (1, 0), (1.2, 0.5), (1, 1), (0.5, 0.8), (0, 1), (-0.2, 0.5)]
enlargement_factor_complex2 = 0.85
enlarged_complex2 = enlarge_polygon(points_complex2, enlargement_factor_complex2)

plot_polygon(points_complex2, "Original Complex", "purple")
plot_polygon(enlarged_complex2, "Shrunk Complex", "brown")
plt.axis('equal')
plt.legend()
plt.title("Polygon Shrinking - Complex Shape")
plt.show()