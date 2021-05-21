import matplotlib.pyplot as plt
from sympy import *
import random
import numpy as np

MINX = 0.
MAXX = 10.
MINY = 0.
MAXY = 10.

# total number of points
n = 50


def draw():
    ax.set_xlim([MINX - 1, MAXX + 1])
    ax.set_ylim([MINY - 1, MAXY + 1])
    plt.scatter(list(map(lambda x: x[0], points)), list(map(lambda x: x[1], points)), c=colors)


def define_hull(hull):
    draw()
    plt.plot(list(map(lambda x: x[0], hull)), list(map(lambda x: x[1], hull)), 'bo-')


def generate_points():
    generated_points = []
    for i in range(n):
        generated_points.append([random.uniform(MINX, MAXX), random.uniform(MINY, MAXY)])
    return generated_points


def distance_to_segment(point, segment):
    return Segment(*segment).distance(Point(point)).evalf()


def is_left_turn(start, end, point):
    x_1 = end[0] - start[0]
    y_1 = end[1] - start[1]
    x_2 = point[0] - start[0]
    y_2 = point[1] - start[1]
    if x_1 * y_2 - x_2 * y_1 > 0:
        return True
    else:
        return False


def rec_fh(current_points, segment):
    if len(current_points) == 0:
        return []
    farmost_point = max(current_points,
                        key=lambda cur: distance_to_segment(cur, segment))
    colors[points.index(farmost_point)] = "b"

    left_points = []
    right_points = []
    for point in points:
        if point != farmost_point:
            if is_left_turn(segment[0], farmost_point, point):
                left_points.append(point)
            if is_left_turn(farmost_point, segment[1], point):
                right_points.append(point)

    return rec_fh(left_points, (segment[0], farmost_point)) + \
           [farmost_point] + \
           rec_fh(right_points, (farmost_point, segment[1]))


def fast_hull(points):
    leftmost = min(points)
    rightmost = max(points)
    left_points = []
    right_points = []
    for point in points:
        if point != leftmost and point != rightmost:
            if is_left_turn(leftmost, rightmost, point):
                left_points.append(point)
            if is_left_turn(rightmost, leftmost, point):
                right_points.append(point)

    return [leftmost] + rec_fh(left_points, (leftmost, rightmost)) + \
           [rightmost] + rec_fh(right_points, (rightmost, leftmost))


# find the a & b points
def get_bezier_coef(points):
    # since the formulas work given that we have n+1 points
    # then n must be this:
    n = len(points) - 1

    # build coefficents matrix
    C = 4 * np.identity(n)
    np.fill_diagonal(C[1:], 1)
    np.fill_diagonal(C[:, 1:], 1)
    C[0, 0] = 2
    C[n - 1, n - 1] = 7
    C[n - 1, n - 2] = 2

    # build points vector
    P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
    P[0] = points[0] + 2 * points[1]
    P[n - 1] = 8 * points[n - 1] + points[n]

    # solve system, find a & b
    A = np.linalg.solve(C, P)
    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B


# returns the general Bezier cubic formula given 4 control points
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + \
                     3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = get_bezier_coef(points)
    return [
        get_cubic(points[i], A[i], B[i], points[i + 1])
        for i in range(len(points) - 1)
    ]


# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])


if __name__ == '__main__':
    points = generate_points()  # random generation
    colors = ["r"] * len(points)  # points are red. Hull points are blue.
    fig, ax = plt.subplots()
    hull = fast_hull(points)

    # use 50 points between each consecutive points to draw the curve
    hull.append(hull[0])
    approx_points = np.array(hull)
    print(approx_points)
    path = evaluate_bezier(approx_points, 50)
    px, py = path[:, 0], path[:, 1]

    define_hull(hull)
    plt.plot(px, py, 'g-')
    plt.show()
