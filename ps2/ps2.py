### Task 1.1: State representation

### Task 1.2: Initial and goal states

### Task 1.3: State transitions

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
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_3():
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

### Task 1.4: Evaluation function

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
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_4():
    cities = 4
    distances = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]
    
    
    route_1 = evaluation_func(cities, distances, [0, 1, 2, 3])
    route_2 = evaluation_func(cities, distances, [2, 1, 3, 0])
    route_3 = evaluation_func(cities, distances, [1, 3, 2, 0])
    route_4 = evaluation_func(cities, distances, [2, 3, 0, 1])
    
    assert route_1 == route_2
    assert route_1 > route_3
    assert route_1 == route_4, "Have you considered the cost to travel from the last city to the headquarter (first)?"

### Task 1.5: Explain your evaluation function

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


### Task 1.6: Improved hill-climbing

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
    raise NotImplementedError
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

### Task 1.7: Comparison between local search and other search algorithms

"""
Run this cell before you start!
"""

import copy
from enum import Enum
from typing import Callable, Union

class Player(Enum):
    BLACK = 'black'
    WHITE = 'white'

    # returns the opponent of the current player
    def get_opponent(self):
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK
        
# board row and column -> these are constant
ROW, COL = 6, 6
INF = 90129012
WIN = 21092109
MOVE_NONE = (-1, -1), (-1, -1)
TIME_LIMIT = 10

Score = Union[int, float]
Move = tuple[tuple[int, int], tuple[int, int]]
Board = list[list[str]]
State = tuple[Board, Player]
Action = tuple[tuple[int, int], tuple[int, int]]


# prints out the current state of the board in a comprehensible way
def print_state(board: Board) -> None:
    horizontal_rule = "+" + ("-" * 5 + "+") * COL
    for row in board:
        print(horizontal_rule)
        print(f"|  {'  |  '.join(' ' if tile == '_' else tile for tile in row)}  |")
    print(horizontal_rule)


def heuristic(state: State) -> Score:
    """
    Returns the score of the current position.

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
    representing black pawn, white pawn, and empty cell, respectively.

    Returns
    -------
    An evaluation (as a Score).
    """
    bcount = 0
    wcount = 0
    board = state[0]
    for r, row in enumerate(board):
        for tile in row:
            if tile == "B":
                if r == 5:
                    return WIN
                bcount += 1
            elif tile == "W":
                if r == 0:
                    return -WIN
                wcount += 1
    if wcount == 0:
        return WIN
    if bcount == 0:
        return -WIN
    return bcount - wcount

### Task 2.1: Implementing Breakthrough for Minimax

def valid_actions(
    state: State
) -> set[Action]:
    """
    Generates a list containing all possible actions in a particular position for the current player
    to move. Return an empty set if there are no possible actions.

    Parameters
    ----------
    state: A tuple conntaining board and current_player information. Board is a 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. current_player is the colour of the current player to move.

    Returns
    -------
    A set of Actions.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def transition(
        state: State,
        action: Action,
    ) -> State:
    """
    Updates the board configuration by modifying existing values if in_place is set to True,
    or creating a new board with updated values if in_place is set to False.

    Parameters
    ----------
    state: A tuple conntaining board and current_player information. Board is a 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. current_player is the colour of the current player to move.
    action: A tuple containing source and destination position of the pawn.

    Returns
    -------
    The new State.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

# checks if it is a terminal state
def is_terminal(state: State) -> bool:
    """
    Returns True if game is over.

    Parameters
    ----------
    state: A tuple conntaining board and current_player information. Board is a 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. current_player is the colour of the current player to move.

    Returns
    -------
    A bool representing whether the game is over.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def utility(state: State) -> int:
    """
    Returns score of the terminal state from the point of view of black.

    Parameters
    ----------
    state: A tuple conntaining board and current_player information. Board is a 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. current_player is the colour of the current player to move.

    Returns
    -------
    int representing the score.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """


def evaluate (state: State) -> int:
    """
    Returns the value of heuristic(state) if it has hit max_depth, otherwise calls utility(state) if it is a terminal state

    Parameters
    ----------
    state: A tuple conntaining board and current_player information. Board is a 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. current_player is the colour of the current player to move.

    Returns
    -------
    int representing the score.
    """
    #It has hit max_depth
    if not is_terminal(state):
        return heuristic(state)
    
    return utility(state)


def minimax(
    board: Board,
    depth: int,
    max_depth: int,
    current_player: Player,
) -> tuple[Score, Action]:
    """
    Finds the best move for the current player and corresponding evaluation from black's
    perspective for the input board state. Return MOVE_NONE if no move is possible
    (e.g. when the game is over).

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. Your function may modify
        the board internally, but the original board passed as an argument must remain unchanged.

    depth: int, the depth to search for the best move. When this is equal to `max_depth`, you
        should get the evaluation of the position using the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    current_player: Player, the colour of the current player to move.

    Returns
    -------
    A tuple (evaluation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that the current player to move can achieve.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    """

    state = (board,current_player)
    
    if depth == max_depth or is_terminal(state):
        return evaluate(state), MOVE_NONE

    if current_player == Player.BLACK:
        best_score = -INF
    else:
        best_score = INF

    best_move = MOVE_NONE
    next_player = current_player.get_opponent()

    for action in valid_actions(state):
        child = transition(state, action)
        new_board = child[0]
        score = minimax(new_board, depth + 1, max_depth, next_player)[0]

        if current_player == Player.BLACK:
            if score > best_score:
                best_score = score
                best_move = action
        else:
            if score < best_score:
                best_score = score
                best_move = action

    return best_score, best_move

def test_task_2_1():
    # Test cases for valid_actions
    def test_valid_actions():
        # Test case 1: A single black pawn in the middle with all three forward moves possible
        board_1 = [
            list("______"),
            list("__B___"),
            list("______"),
            list("______"),
            list("______"),
            list("______"),
        ]
        state_1 = (board_1, Player.BLACK)
        actions_1 = valid_actions(state_1)
        expected_actions_1 = {
            ((1, 2), (2, 1)), 
            ((1, 2), (2, 2)), 
            ((1, 2), (2, 3))
        }
        assert actions_1 == expected_actions_1, "valid_actions failed for a single black pawn with 3 empty forward squares."
    
        # Test case 2: A single white pawn at the edge with two forward moves possible
        board_2 = [
            list("______"),
            list("______"),
            list("______"),
            list("______"),
            list("W_____"),
            list("______"),
        ]
        state_2 = (board_2, Player.WHITE)
        actions_2 = valid_actions(state_2)
        expected_actions_2 = {
            ((4, 0), (3, 0)), 
            ((4, 0), (3, 1))
        }
        assert actions_2 == expected_actions_2, "valid_actions failed for a single white pawn at the edge."
    
        # Test case 3: Mixed moves for a black pawn
        board_3 = [
            list("______"),
            list("___B__"),
            list("__W_W_"),
            list("______"),
            list("______"),
            list("______"),
        ]
        state_3 = (board_3, Player.BLACK)
        actions_3 = valid_actions(state_3)
        expected_actions_3 = {
            ((1, 3), (2, 2)),
            ((1, 3), (2, 3)),
            ((1, 3), (2, 4))
        }
        assert actions_3 == expected_actions_3, "valid_actions failed for Black's mixed moves (move and capture)."
    
        # Test case 4: A fully blocked black pawn has no moves.
        board_4 = [
            list("______"),
            list("__B___"),
            list("_BWB__"),
            list("______"),
            list("______"),
            list("______"),
        ]
        state_4 = (board_4, Player.BLACK)
        actions_4 = valid_actions(state_4)
        pawn_at_1_2_moves = {action for action in actions_4 if action[0] == (1, 2)}
        assert len(pawn_at_1_2_moves) == 0, "valid_actions failed for a fully blocked pawn; it should have no moves."
    
        print("All valid_actions tests passed!")
    
    # Test cases for transition
    def test_transition():
        # Test case 1: Black pawn moves forward
        board_1 = [
            list("______"),
            list("__B___"),
            list("______"),
            list("______"),
            list("______"),
            list("______"),
        ]
        state_1 = (board_1, Player.BLACK)
        action_1 = ((1, 2), (2, 2))
        new_state_1 = transition(state_1, action_1)
        expected_board_1 = [
            list("______"),
            list("______"),
            list("__B___"),
            list("______"),
            list("______"),
            list("______"),
        ]
        assert new_state_1[0] == expected_board_1, "transition failed for Black's forward move."
    
        # Test case 2: White pawn captures a black pawn
        board_2 = [
            list("______"),
            list("__B___"),
            list("_W____"),
            list("______"),
            list("______"),
            list("______"),
        ]
        state_2 = (board_2, Player.WHITE)
        action_2 = ((2, 1), (1, 2))
        new_state_2 = transition(state_2, action_2)
        expected_board_2 = [
            list("______"),
            list("__W___"),
            list("______"),
            list("______"),
            list("______"),
            list("______"),
        ]
        assert new_state_2[0] == expected_board_2, "transition failed for White's capture move."
        
        # Test case 3: Original board is not modified
        original_board = [
            list("______"),
            list("__B___"),
            list("______"),
            list("______"),
            list("______"),
            list("______"),
        ]
        original_board_copy = copy.deepcopy(original_board)
        state_3 = (original_board, Player.BLACK)
        action_3 = ((1, 2), (2, 2))
        transition(state_3, action_3)
        assert original_board == original_board_copy, "transition function should not modify the original board."
    
        print("All transition tests passed!")
    
    # Test cases for is_terminal
    def test_is_terminal():
        # Test case 1: Black wins by reaching the end
        board_1 = [
            list("______"), list("______"), list("______"),
            list("______"), list("______"), list("___B__"),
        ]
        assert is_terminal((board_1, Player.BLACK)) is True, "is_terminal failed for Black's win condition."
    
        # Test case 2: White wins by eliminating all black pieces
        board_2 = [
            list("______"), list("__W___"), list("______"),
            list("______"), list("______"), list("___W__"),
        ]
        assert is_terminal((board_2, Player.WHITE)) is True, "is_terminal failed when Black has no pieces."
    
        # Test case 3: Non-terminal state
        board_3 = [
            list("______"), 
            list("__B___"), 
            list("______"),
            list("_W____"), 
            list("______"), 
            list("______"),
        ]
        assert is_terminal((board_3, Player.BLACK)) is False, "is_terminal failed for a non-terminal state."
    
        print("All is_terminal tests passed!")
    
    # Test cases for utility
    def test_utility():
        # Test case 1: Black wins
        board_1 = [
            list("______"), list("______"), list("______"),
            list("______"), list("______"), list("B_____"),
        ]
        assert utility((board_1, Player.BLACK)) == WIN, "utility failed for Black's win."
    
        # Test case 2: White wins
        board_2 = [
            list("W_____"), list("______"), list("______"),
            list("______"), list("______"), list("______"),
        ]
        assert utility((board_2, Player.WHITE)) == -WIN, "utility failed for White's win."
    
        # Test case 3: Black wins (no white pieces)
        board_3 = [
            list("______"), list("B_____"), list("______"),
            list("______"), list("______"), list("______"),
        ]
        assert utility((board_3, Player.BLACK)) == WIN, "utility failed for Black's win by eliminating opponent."
    
        print("All utility tests passed!")
    
    test_valid_actions()
    test_transition()
    test_is_terminal()
    test_utility()
    
    print("\nAll new test cases for core functions passed successfully!")

### Task 2.2: Integrate alpha-beta pruning into minimax

def minimax_alpha_beta(
    board: Board,
    depth: int,
    max_depth: int,
    alpha: Score,
    beta: Score,
    current_player: Player
) -> tuple[Score, Move]:
    """
    Finds the best move for the current player and corresponding evaluation from black's
    perspective for the input board state. Return MOVE_NONE if no move is possible
    (e.g. when the game is over).

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. Your function may modify
        the board internally, but the original board passed as an argument must remain unchanged.

    depth: int, the depth to search for the best move. When this is equal to `max_depth`, you
        should get the evaluation of the position using the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    alpha: Score. The alpha value in a given state.

    beta: Score. The beta value in a given state.

    current_player: Player, the colour of the current player
        to move.

    Returns
    -------
    A tuple (evaluation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that the current player to move can achieve.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

board_41 = [
    list("______"),
    list("__BB__"),
    list("____BB"),
    list("WBW_B_"),
    list("____WW"),
    list("_WW___"),
]

board_42 = [
    list("____B_"),
    list("__BB__"),
    list("______"),
    list("_WWW__"),
    list("____W_"),
    list("______"),
]

def test_task_2_2():
    test_board_preservation(minimax_alpha_beta)
    test_game_over(minimax_alpha_beta, game_over_board_1, -WIN)
    test_game_over(minimax_alpha_beta, game_over_board_2, WIN)
    test_max_depth(minimax_alpha_beta, max_depth_board_1, Player.BLACK, [((3, 5), (4, 4)), ((3, 5), (4, 5))])
    test_max_depth(minimax_alpha_beta, max_depth_board_2, Player.WHITE, [((2, 0), (1, 0)), ((2, 0), (1, 1))])
    test_player_switching(minimax_alpha_beta, Player.BLACK)
    test_player_switching(minimax_alpha_beta, Player.WHITE)
    
    test_search(minimax_alpha_beta, board_41, 3, Player.BLACK, WIN, [((3, 4), (4, 5))])
    test_search(minimax_alpha_beta, board_42, 6, Player.BLACK, -WIN, [((0, 4), (1, 4)), ((0, 4), (1, 5)), ((1, 2), (2, 2)), ((1, 2), (2, 1)), ((1, 2), (2, 3)), ((1, 3), (2, 3)), ((1, 3), (2, 2)), ((1, 3), (2, 4))])
    test_search(minimax_alpha_beta, board_33, 3, Player.WHITE, -WIN, [((2, 0), (1, 0)), ((2, 0), (1, 1))])
    test_search(minimax_alpha_beta, board_34, 3, Player.WHITE, -1, [((3, 1), (2, 2)), ((4, 2), (3, 3)), ((4, 4), (3, 3)), ((4, 4), (3, 5))])

### Task 2.3: Implement an improved heuristic function

def improved_evaluate(state: State) -> Score:
    """
    Returns the score of the current position with an improved heuristic.

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively.

    Returns
    -------
    An improved evaluation (as a Score).
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_3():
    board_51 = [
        list("___B__"),
        list("______"),
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
    ]
    
    board_52 = [
        list("___BW_"),
        list("___W__"),
        list("______"),
        list("______"),
        list("______"),
        list("______"),
    ]
    
    board_53 = [
        list("______"),
        list("______"),
        list("______"),
        list("__B___"),
        list("______"),
        list("______"),
    ]
    
    board_54 = [
        list("__B___"),
        list("__WB__"),
        list("______"),
        list("______"),
        list("_____"),
        list("______"),
    ]
    
    assert improved_evaluate((board_51, Player.BLACK)) > heuristic((board_51, Player.BLACK)), "Your improved evaluation function should return higher than the original heuristic, as black is winning."
    assert improved_evaluate((board_52, Player.BLACK)) == -WIN, "Your improved evaluation function does not correctly evaluate won positions."
    assert improved_evaluate((board_53, Player.BLACK)) == WIN, "Your improved evaluation function does not correctly evaluate won positions."
    assert improved_evaluate((board_54, Player.BLACK)) < heuristic((board_54, Player.BLACK)), "Your improved evaluation function should return smaller than the original heuristic, as white is winning."


if __name__ == '__main__':
    test_task_1_3()
    test_task_1_4()
    test_task_1_6()
    test_task_2_1()
    test_task_2_2()
    test_task_2_3()