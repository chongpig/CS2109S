def uninformed_search(problem: cube.Cube):
    r"""
    Uninformed Search finds the solution to reach the goal from the initial.
    If no solution is found, return False.
    
    Args:
        problem (cube.Cube): Cube instance

    Returns:
        solution (List[Action]): the action sequence
    """
    """ YOUR CODE HERE """
    def heuristicZero(problem: cube.Cube, state: cube.State) -> float:
        return 0
    return astar_search(problem, heuristicZero)
    """ YOUR CODE END HERE """
