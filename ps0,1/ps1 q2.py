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
    return 2*(m+1)*(c+1)
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
    if ml == m and cl == c and p == 1:
        return True
    else:
        return False
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
    actions = [(1, 0), (1, 1), (2, 0), (0, 1), (0, 2)]
    return actions
                
        
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
    ml,cl,p = state
    mMove, cMove = action
    if p == 0:
        mlNew = ml + mMove
        clNew = cl + cMove
        pNew = 1
    if p == 1:
        mlNew = ml - mMove
        clNew = cl - cMove
        pNew = 0
    return (mlNew,clNew,pNew)
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
    queue = []
    initial_state = get_initial_state(m, c)
    queue.append((initial_state, ()))
    length = get_max_steps(m,c)
    visited = set()
    visited.add(initial_state)
    while queue:
        state, steps = queue.pop(0)
        if len(steps) > length :
            continue
        if is_goal(m, c, state):
            print (steps)
            return steps
        else:
            for a in valid_actions(state):
                next_state = transition(state, a)
                ml, cl, p = next_state
                if ml < 0 or cl < 0 or ml > m or cl > c:
                    continue
                if ml > 0 and ml < cl:
                    continue
                mr = m - ml
                cr = c - cl
                if mr > 0 and mr < cr:
                    continue
                if next_state not in visited:
                    visited.add(next_state)
                    next_steps = steps + (a,)
                    queue.append((next_state, next_steps))
    return False
mnc_search(3, 3)
