def expand(problem, node): # Generate children nodes to be explored
    for act in problem.actions(node.state):
        state = problem.result(node.state, act)
        current_cost = problem.path_cost(node.cost, node.state, act, state)
        child = Node(parent=node, 
                    act=act, 
                    state=state, 
                    cost=current_cost)
        yield child

def astar_search(problem: cube.Cube, heuristic_func: Callable):
    r"""
    A* Search finds the solution to reach the goal from the initial.
    If no solution is found, return False.
    
    Args:
        problem (cube.Cube): Cube instance
        heuristic_func (Callable): heuristic function for the A* search

    Returns:
        solution (List[Action]): the action sequence
    """
    fail = True
    solution = []

    reached = set()
    frontier = PriorityQueue()
    initial = problem.initial
    curr = Node(parent=None, 
                act=None, 
                state=initial,
                cost=0)
    
    """ YOUR CODE HERE """
    frontier.push(heuristic_func(problem, curr.state),curr)
    reached.add(initial)
    while frontier:
        curr = frontier.pop()
        if problem.goal_test(curr.state) == True:
            fail = False
            break
        children = expand(problem, curr)
        for c in children:
            if not(c.state in reached) :
                reached.add(c.state)
                fc = c.cost + heuristic_func(problem, c.state)
                frontier.push(fc,c)
    """ YOUR CODE END HERE """

    if not fail:
        while curr.parent:
            solution.append(curr.act)
            curr = curr.parent
    solution.reverse()

    if fail:
        return False
    return solution
