# Improved version of your wind farm layout optimization script using multiprocessing
# for dual-core processors, including original plots

import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# Load site and turbine
site = Hornsrev1Site()
wt = V80()
windFarmModel = Bastankhah_PorteAgel_2014(site, wt, k=0.0324555)

wt_x, wt_y = site.initial_position.T.tolist()
WT_Num = len(wt_x)
aep_ref = round(float(windFarmModel(wt_x, wt_y).aep().sum()), 4)
WT_Rad = 40
Iter_Length = 100
Iter_Num = 10
Gen_Num = 5
Border_Margin = 1.1

# Fitness function
def fitness(s):
    penalty = 0
    for i in range(WT_Num):
        x1, y1 = s[0][i], s[1][i]
        for j in range(i + 1, WT_Num):
            x2, y2 = s[0][j], s[1][j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance <= 5 * 2 * WT_Rad:
                penalty += -abs(distance - 5 * 2 * WT_Rad) * 100
    return float(windFarmModel(s[0], s[1]).aep().sum()) * 1e6 + penalty

# Geometry helpers
def select_border_points(points):
    hull = ConvexHull(points)
    return points[hull.vertices]

def enlarge_polygon(points, factor):
    center = np.mean(points, axis=0)
    return np.array([center + (p - center) * factor for p in points])

def create_mesh_border(points):
    codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)
    return Path(points, codes)

def filter_points_inside_border(border_path, point):
    return border_path.contains_point((point[0], point[1]))

# Random layout modifier
def list_random_values_one_by_one(lst1, lst2, length, border):
    n = len(lst1)
    lst1_rnd, lst2_rnd = lst1[:], lst2[:]
    current_fitness = fitness((lst1_rnd, lst2_rnd))

    for i in range(n):
        bestx, besty = lst1_rnd[:], lst2_rnd[:]
        for _ in range(3):
            while True:
                newx = bestx[i] + length * np.random.uniform(-1, 1)
                newy = besty[i] + length * np.random.uniform(-1, 1)
                if filter_points_inside_border(border, (newx, newy)):
                    break
            trialx, trialy = bestx[:], besty[:]
            trialx[i], trialy[i] = newx, newy
            if fitness((trialx, trialy)) > fitness((bestx, besty)):
                bestx, besty = trialx, trialy

        if fitness((bestx, besty)) >= current_fitness:
            lst1_rnd, lst2_rnd = bestx, besty
            current_fitness = fitness((lst1_rnd, lst2_rnd))

    return lst1_rnd, lst2_rnd

def single_solution_evaluation(args):
    return list_random_values_one_by_one(*args)

# Multiprocessing solution generator
def getSolutions():
    points = np.array(list(zip(wt_x, wt_y)))
    border_points = enlarge_polygon(select_border_points(points), Border_Margin)
    mesh_border = create_mesh_border(border_points)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(single_solution_evaluation, [(wt_x, wt_y, Iter_Length, mesh_border) for _ in range(Iter_Num)])

    ranked = sorted([(fitness(sol), sol) for sol in results], key=lambda x: x[0])
    best_fitness, best_coords = ranked[-1]
    x_best, y_best = best_coords
    aep_best = round(float(windFarmModel(x_best, y_best).aep().sum()), 4)

    return (wt_x, wt_y) if aep_best < aep_ref else (x_best, y_best), max(aep_ref, aep_best), ranked, border_points

# Genetic solution optimizer
def getGenSolutions(initial, aep_best_prev, border_points):
    NewGen = initial
    Gen = []
    GenBest = []
    aep_best_series = []

    for i in range(1, Gen_Num + 1):
        Gen.append(i)
        mesh_border = create_mesh_border(border_points)

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(single_solution_evaluation, [(NewGen[0], NewGen[1], Iter_Length, mesh_border)] * Iter_Num)

        ranked = sorted([(fitness(sol), sol) for sol in results], key=lambda x: x[0])
        best_fitness, best_coords = ranked[-1]
        aep = round(float(windFarmModel(best_coords[0], best_coords[1]).aep().sum()), 4)

        if aep > aep_best_prev:
            NewGen = best_coords
            aep_best_prev = aep

        GenBest.append(NewGen)
        aep_best_series.append(aep_best_prev)

        print(f"Gen {i}: AEP = {aep_best_prev} GWh")

    return NewGen, Gen, GenBest, aep_best_series

if __name__ == '__main__':
    best_sol, aep_start, ranked, border_points = getSolutions()
    final_sol, gens, gens_best, aep_list = getGenSolutions(best_sol, aep_start, border_points)

    print("Final Layout AEP:", round(float(windFarmModel(final_sol[0], final_sol[1]).aep().sum()), 4), "GWh")
    
    # Plot 1 - Original and Genetic Layouts
    plt.figure()
    plt.title('Original blue and evolutions of Generation Layouts yellow')
    plt.plot(wt_x, wt_y, 'b.')
    for s in gens_best:
        plt.plot(s[0], s[1], 'y.')
    plt.plot(final_sol[0], final_sol[1], 'g.')
    plt.scatter(border_points[:, 0], border_points[:, 1], c='r', label="Border Points")
    plt.plot(border_points[:, 0], border_points[:, 1], 'r--')
    plt.plot([border_points[-1, 0], border_points[0, 0]], [border_points[-1, 1], border_points[0, 1]], 'r--')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()

    # Plot 2 - Initial vs Final
    plt.figure()
    plt.title('Original blue and Final black Layouts')
    plt.plot(wt_x, wt_y, 'b.')
    plt.plot(final_sol[0], final_sol[1], 'k.')
    plt.scatter(border_points[:, 0], border_points[:, 1], c='r', label="Border Points")
    plt.plot(border_points[:, 0], border_points[:, 1], 'r--')
    plt.plot([border_points[-1, 0], border_points[0, 0]], [border_points[-1, 1], border_points[0, 1]], 'r--')
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # Plot 3 - AEP evolution
    plt.figure()
    plt.title('Evolution of improvement through generations')
    x = gens
    y = aep_list
    coef = np.polyfit(x, y, 2)
    parabola = np.poly1d(coef)
    x_trend = np.linspace(min(x), max(x), 100)
    y_trend = parabola(x_trend)
    plt.scatter(x, y, label='AEP Data Points')
    plt.plot(x_trend, y_trend, 'r-', label='Parabolic Trend Line')
    plt.xlabel('Generation')
    plt.ylabel('AEP (GWh)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 4 - Delta AEP evolution
    delta = np.diff([aep_ref] + aep_list)
    plt.figure()
    plt.title('Evolution of AEP improvements per generation')
    coef = np.polyfit(gens, delta, 2)
    parabola = np.poly1d(coef)
    y_trend = parabola(x_trend)
    plt.scatter(gens, delta, label='Delta AEP Data Points')
    plt.plot(x_trend, y_trend, 'r-', label='Parabolic Trend Line')
    plt.xlabel('Generation')
    plt.ylabel('Delta AEP (GWh)')
    plt.legend()
    plt.grid(True)
    plt.show()
   