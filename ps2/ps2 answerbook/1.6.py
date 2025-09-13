"""
Run this cell before you start!
"""

import random
import time

from typing import List, Tuple, Callable

def hill_climbing(
    cities: int,
    distances: List[Tuple[int]],
    successor: Callable,
    evaluation_func: Callable
) -> List[int]:
    """
    Hill climbing finds the solution to reach the goal from the initial.

    Args:
        cities (int): The number of cities to be visited.

        distances (List[Tuple[int]]): The list of distances between every two cities. Each distance
            is represented as a tuple in the form of (c1, c2, d), where c1 and c2 are the two cities
            and d is the distance between them. The length of the list should be equal to cities *
            (cities - 1)/2.

        Successor (Callable): A function that generates new routes to be used in the next
            iteration in the hill-climbing algorithm. Will be provided on Coursemology.

        evaluation_func (Callable): A function that computes the evaluation score of a route. Will
            be provided on Coursemology.

    Returns:
        route (List[int]): The shortest route, represented by a list of cities in the order to be
            traversed.
    """
    route = random.sample(list(range(cities)), cities)
    curr_hn = evaluation_func(cities, distances, route)
    while True:
        new_routes = successor(route)
        best_new_route = max(new_routes, key=lambda x: evaluation_func(cities, distances, x))
        h_n = evaluation_func(cities, distances, best_new_route)
        if h_n <= curr_hn:
            return route

        curr_hn = h_n
        route = best_new_route

def hill_climbing_improved(
    cities: int,
    distances: List[Tuple[int]],
    successor: Callable,
    evaluation_func: Callable,
    hill_climbing: Callable
) -> List[int]:
    """
    Improved hill climbing that finds the solution to reach the goal from the initial.

    Args:
        cities (int): The number of cities to be visited.

        distances (List[Tuple[int]]): The list of distances between every two cities. Each distance
            is represented as a tuple in the form of (c1, c2, d), where c1 and c2 are the two cities
            and d is the distance between them. The length of the list should be equal to cities *
            (cities - 1)/2.

        successor (Callable): The successor function to be used in hill climbing. Will be
            provided on Coursemology.

        evaluation_func (Callable): The evaluation function to be used in hill climbing. Will be
            provided on Coursemology.

        hill_climbing (Callable): The hill climbing function to be used for each restart. Will be
            provided on Coursemology.

    Returns:
        route (List[int]): The shortest route, represented by a list of cities in the order to be
            traversed.
    """
    """ YOUR CODE HERE """
    bestValue = - 99999999
    for i in range(10):
        tempRoute = hill_climbing(cities, distances, successor, evaluation_func)
        value = evaluation_func(cities, distances, tempRoute)
        if value > bestValue:
            bestRoute = tempRoute
            bestValue = value
    return bestRoute            
    """ YOUR CODE END HERE """
    
def test_task_1_6():
    def test_improved_hill_climbing(cities: int, distances: List[Tuple[int]], successor, evaluation_func, hill_climbing, hill_climbing_improved):
        route = hill_climbing_improved(cities, distances, successor, evaluation_func, hill_climbing)
        assert sorted(route) == list(range(cities)), "New route does not contain all cities present in the original route."
    
    cities_1 = 4
    distances_1 = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]
    
    test_improved_hill_climbing(cities_1, distances_1, successor, evaluation_func, hill_climbing, hill_climbing_improved)
    
    cities_2 = 10
    distances_2 = [(2, 7, 60), (1, 6, 20), (5, 4, 70), (9, 8, 90), (3, 7, 54), (2, 5, 61),
        (4, 1, 106), (0, 6, 51), (3, 1, 45), (0, 5, 86), (9, 2, 73), (8, 4, 14), (0, 1, 51),
        (9, 7, 22), (3, 2, 22), (8, 1, 120), (5, 7, 92), (5, 6, 60), (6, 2, 10), (8, 3, 78),
        (9, 6, 82), (0, 2, 41), (2, 8, 99), (7, 8, 71), (0, 9, 32), (4, 0, 73), (0, 3, 42),
        (9, 1, 80), (4, 2, 85), (5, 9, 113), (3, 6, 28), (5, 8, 81), (3, 9, 72), (9, 4, 81),
        (5, 3, 45), (7, 4, 60), (6, 8, 106), (0, 8, 85), (4, 6, 92), (7, 6, 70), (7, 0, 22),
        (7, 1, 73), (4, 3, 64), (5, 1, 80), (2, 1, 22)]
    
    test_improved_hill_climbing(cities_2, distances_2, successor, evaluation_func, hill_climbing, hill_climbing_improved)
