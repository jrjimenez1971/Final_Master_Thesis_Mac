# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 23:07:27 2025

@author: jrjim
"""

import numpy as np
from scipy.spatial import ConvexHull

def select_border_points(points):
    """
    Selects the border points of a list of points enclosed in an area using Convex Hull.

    Args:
        points: A list or NumPy array of (x, y) coordinates.

    Returns:
        A NumPy array of border points.
    """

    points = np.array(points)  # Ensure points is a NumPy array
    if points.shape[0] < 3:
        return points #a line or point, return points.

    hull = ConvexHull(points)
    border_points = points[hull.vertices]
    return border_points

# Example usage:
if __name__ == "__main__":
    points = [
        (1, 1), (2, 2), (3, 1), (2, 0), (1.5, 1.5), (2.5, 1.5), (1.8, 0.8), (2.2, 0.8), (3.5,2), (0.5,2)
    ]

    border_points = select_border_points(points)
    print("Border Points:")
    print(border_points)

    # Visualization (optional):
    import matplotlib.pyplot as plt

    plt.scatter([p[0] for p in points], [p[1] for p in points], label="All Points")
    plt.scatter(border_points[:, 0], border_points[:, 1], c='r', label="Border Points")
    plt.plot(border_points[:,0], border_points[:,1], 'r--', lw=1)
    plt.plot([border_points[-1,0], border_points[0,0]],[border_points[-1,1], border_points[0,1]], 'r--', lw=1)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Border Points Selection")
    plt.grid(True)
    plt.axis('equal')
    plt.show()