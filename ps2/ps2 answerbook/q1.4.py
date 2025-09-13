"""
Run this cell before you start!
"""

import random
import time

from typing import List, Tuple, Callable


def evaluation_func(
    cities: int,
    distances: List[Tuple[int]],
    route: List[int]
) -> float:
    """
    Computes the evaluation score of a route

    Args:
        cities (int): The number of cities to be visited.

        distances (List[Tuple[int]]): The list of distances between every two cities. Each distance
            is represented as a tuple in the form of (c1, c2, d), where c1 and c2 are the two cities
            and d is the distance between them. The length of the list should be equal to cities *
            (cities - 1)/2.

        route (List[int]): The current route as a list of cities in the order of travel.

    Returns:
        h_n (float): the evaluation score.
    """
    """ YOUR CODE HERE """
    distMap = {}
    for c1, c2, d in distances:
        distMap[(c1, c2)] = d
        distMap[(c2, c1)] = d
    totalDistance = 0
    for i in range(len(route)):
        c1 = route[i]
        c2 = route[(i + 1) % len(route)]
        totalDistance += distMap[(c1, c2)]
    return -totalDistance
    """ YOUR CODE END HERE """

# Test cases
cities = 4
distances = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]


route_1 = evaluation_func(cities, distances, [0, 1, 2, 3])
route_2 = evaluation_func(cities, distances, [2, 1, 3, 0])
route_3 = evaluation_func(cities, distances, [1, 3, 2, 0])
route_4 = evaluation_func(cities, distances, [2, 3, 0, 1])

assert route_1 == route_2
assert route_1 > route_3
assert route_1 == route_4, "Have you considered the cost to travel from the last city to the headquarter (first)?"
