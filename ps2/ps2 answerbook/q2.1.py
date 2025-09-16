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
    board, currentPlayer = state
    actions = set() 
    if currentPlayer == Player.BLACK:
        dr = 1  
    else:
        dr = -1
    piece = 'B' 
    if currentPlayer == Player.BLACK:
        piece = 'B' 
    else:
        piece = 'W'
    if currentPlayer == Player.BLACK:
        opponent = 'W'
    else:
        opponent = 'B'
    for r in range(ROW):
        for c in range(COL):
            if board[r][c] == piece:
                for dc in [-1, 0, 1]:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < ROW and 0 <= nc < COL:
                        target = board[nr][nc]
                        if dc == 0:
                            if target == '_':
                                actions.add(((r, c), (nr, nc)))
                        else:
                            if target == '_' or target == opponent:
                                actions.add(((r, c), (nr, nc)))
    return actions
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
    board, currentPlayer = state
    newBoard = copy.deepcopy(board)
    (sr, sc), (dr, dc) = action
    newBoard[dr][dc] = newBoard[sr][sc]
    newBoard[sr][sc] = '_'
    nextPlayer = currentPlayer.get_opponent()
    return (newBoard, nextPlayer)
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
    board, _ = state
    bcount = 0
    wcount = 0
    blackIn5 = False
    whiteIn0 = False
    for r in range(ROW):
        for c in range(COL):
            if board[r][c] == 'B':
                bcount += 1
                if r == 5:
                    blackIn_5 = True
            elif board[r][c] == 'W':
                wcount += 1
                if r == 0:
                    whiteIn0 = True
    return blackIn5 or wcount == 0 or whiteIn0 or bcount == 0
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
    board, _ = state
    bcount = 0
    wcount = 0
    blackWin = False
    whiteWin = False
    for r in range(ROW):
        for c in range(COL):
            if board[r][c] == 'B':
                bcount += 1
                if r == 5:
                    blackWin = True
            elif board[r][c] == 'W':
                wcount += 1
                if r == 0:
                    white_win = True
    if blackWin or wcount == 0:
        return WIN
    if white_win or bcount == 0:
        return -WIN
    return 0
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

def minimax_alpha_beta(
    board: Board,
    depth: int,
    max_depth: int,
    alpha: Score,
    beta: Score,
    current_player: Player
) -> tuple[Score, Move]:
    state = (board, current_player)
    if depth == max_depth or is_terminal(state):
        return evaluate(state), MOVE_NONE
    bestMove = MOVE_NONE
    nextPlayer = current_player.get_opponent()
    if current_player == Player.BLACK:
        bestScore = -INF
        for action in valid_actions(state):
            child = transition(state, action)
            newBoard = child[0]
            score = minimax_alpha_beta(newBoard, depth + 1, max_depth, alpha, beta, nextPlayer)[0]
            if score > bestScore:
                bestScore = score
                bestMove = action
            alpha = max(alpha, bestScore)
            if alpha >= beta:
                break
        return bestScore, bestMove
    else:
        bestScore = INF
        for action in valid_actions(state):
            child = transition(state, action)
            newBoard = child[0]
            score = minimax_alpha_beta(newBoard, depth + 1, max_depth, alpha, beta, nextPlayer)[0]
            if score < bestScore:
                bestScore = score
                bestMove = action
            beta = min(beta, bestScore)
            if alpha >= beta:
                break
        return bestScore, bestMove

# Test cases
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

# Test cases
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