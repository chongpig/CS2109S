def get_initial_state(m, c):
    '''
    Create the initial state for the missionaries and cannibals problem.
    
    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Return the starting state derived from `m` and `c`. This could be the state representation you described in task 1.1.
    '''
    """ YOUR CODE HERE """
    return (0, 0, 0)
    """ YOUR CODE END HERE """

def get_max_steps(m, c):
    '''
    Calculate a safe upper bound on the number of moves needed to solve the problem.
    
    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns an integer representing the maximum number of steps to explore before giving up. This could be the upper bound you described in task 1.5.
    '''
    """ YOUR CODE HERE """
    return 2 * (m + 1) * (c + 1)
    """ YOUR CODE END HERE """

def is_goal(m, c, state):
    '''
    Check if the given state is a goal state.
    
    Parameters
    ----------    
    state: current state
    m: number of missionaries
    c: number of cannibals
    
    Returns
    ----------    
    Returns True if the state has reached your proposed goal state.
    '''
    """ YOUR CODE HERE """
    ml, cl, p = state
    return ml == m and cl == c and p == 1
    """ YOUR CODE END HERE """

def valid_actions(state):
    '''
    Generate all valid actions from the current state.
    
    Parameters
    ----------    
    state: current state
    
    Returns
    ----------    
    Returns a set of valid actions which can be performed at the current state
    '''
    """ YOUR CODE HERE """
    return [(1, 0), (1, 1), (2, 0), (0, 1), (0, 2)]
    """ YOUR CODE END HERE """

def transition(state, action):
    '''
    Apply an action to the current state to get the next state.
    
    Parameters
    ----------    
    state: current state
    action: your representation from valid_actions function above for the action to take
    
    Returns
    ----------    
    Returns the state after applying the action.
    '''
    """ YOUR CODE HERE """
    ml, cl, p = state
    mMove, cMove = action
    if p == 0:
        ml_new = ml + mMove
        cl_new = cl + cMove
        p_new = 1
    else:
        ml_new = ml - mMove
        cl_new = cl - cMove
        p_new = 0
    return (ml_new, cl_new, p_new)
    """ YOUR CODE END HERE """

def mnc_search(m, c):  
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    def dfs(state, steps, depth, max_depth):
        if len(steps) > max_depth:
            return None
        if is_goal(m, c, state):
            return steps
        for a in valid_actions(state):
            next_state = transition(state, a)
            if next_state is None:
                continue
            ml, cl, p = next_state
            if ml < 0 or cl < 0 or ml > m or cl > c:
                continue
            if ml > 0 and ml < cl:
                continue
            mr, cr = m - ml, c - cl
            if mr > 0 and mr < cr:
                continue
            next_steps = steps + (a,)
            result = dfs(next_state, next_steps, depth + 1, max_depth)
            if result is not None:
                return result
        return None
    initial_state = get_initial_state(m, c)
    max_steps = get_max_steps(m, c)
    for depth_limit in range(max_steps + 1):
        result = dfs(initial_state, (), 0, depth_limit)
        if result is not None:
            result = result[::-1]
            return result
    return False
