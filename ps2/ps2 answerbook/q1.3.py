"""
Run this cell before you start!
"""

import random
import time

from typing import List, Tuple, Callable

def successor(route: List[int]) -> List[List[int]]:
    """
    Generates new routes to be used in the next iteration in the hill-climbing algorithm.

    Args:
        route (List[int]): The current route as a list of cities in the order of travel.

    Returns:
        new_routes (List[List[int]]): New routes to be considered.
    """
    """ YOUR CODE HERE """
    n = len(route)
    new_routes = []
    for i in range(n - 1):
        new_route = route[:]
        new_route[i], new_route[i + 1] = new_route[i + 1], new_route[i]
        new_routes.append(new_route)
    for skip in [2, 3]:
        for i in range(n - skip):
            new_route = route[:]
            new_route[i], new_route[i+skip] = new_route[i+skip], new_route[i]
            new_routes.append(new_route)
    return new_routes
    """ YOUR CODE END HERE """

    # Test cases
def test_successor(route):
    sorted_route = sorted(route)
    result = successor(route)
    assert result is not None, "Successor function returns an empty list."
    assert any(result), "Successor function returns an empty list."
    for new_route in result:
        assert len(new_route) == len(sorted_route), "New route does not have the same number of cities as the original route."
        assert sorted(new_route) == sorted_route, "New route does not contain all cities present in the original route."

permutation_route = list(range(4))
new_permutation_routes = successor(permutation_route)
assert len(new_permutation_routes) < 24, "Your successor function may have generated too many new routes by enumerating all possible states."

test_successor([1, 3, 2, 0])
test_successor([7, 8, 6, 3, 5, 4, 9, 2, 0, 1])
