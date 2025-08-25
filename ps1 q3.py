def heuristic_func(problem: cube.Cube, state: cube.State) -> float:
    r"""
    Computes the heuristic value of a state
    
    Args:
        problem (cube.Cube): the problem to compute
        state (cube.State): the state to be evaluated
        
    Returns:
        h_n (float): the heuristic value 
    """
    h_n = 0.0
    goals = problem.goal

    """ YOUR CODE HERE """
    current = state.layout
    goal = problem.goal.layout
    rows, cols = state.shape
    moveReq = 0
    for i in range(len(current)):
        if current[i] != goal[i]:
            moveReq += 1
    maxMove = max(rows, cols)
    h_n = math.ceil(moveReq / maxMove)
    """ YOUR CODE END HERE """

    return h_n
