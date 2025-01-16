from utils import *
from game_state import StateStatus, GameState


# the same as CountOwnPieces.evaluate() in heuristics.py
def B_W_counts(state: GameState, color: PlayerColorType) -> float:
    bn, wn = state.black_white_counts
    if color == BLACK:
        return bn  # count of black
    else:
        return wn  # count of white


# stable pieces (i.e. position with chess) that will not be flipped
def is_stable(state: GameState, color: PlayerColorType, i: int, j: int) -> bool:
    grid = state._grid
    opponent_color = -color

    if (i, j) in CORNERS:
        return True

    directions = [(0, 1), (1, -1), (1, 0), (1, 1)]  # four basic directions
    """     
    When positions of certain 2 opposite direction of (x, y) contains the following content:
        1. there's an EMPTY in one direction, and an opponent_color in another;
        2. there's an EMPTY in both directions;
    Then this (x, y) is not stable, it's color could be changed. 
    """
    s1 = ""
    s2 = ""

    for dx, dy in directions:

        x, y = i, j
        for _ in range(BOARD_WIDTH):
            x += dx
            y += dy

            if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_WIDTH):
                break  # Out of bounds

            if grid[x][y] == opponent_color:
                s1 = "op"
            elif grid[x][y] == EMPTY:
                s1 = "empty"
                break

        x, y = i, j
        for _ in range(BOARD_WIDTH):
            x -= dx
            y -= dy

            if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_WIDTH):
                break  # Out of bounds

            if grid[x][y] == opponent_color:
                s2 = "op"
            elif grid[x][y] == EMPTY:
                s2 = "empty"
                break

        if {s1, s2} == {"op", "empty"} or {s1, s2} == {"empty", "empty"}:
            return False

    return True


def n_stables(state: GameState, color: PlayerColorType) -> int:
    # stables = []
    n = 0
    for i in range(BOARD_WIDTH):
        for j in range(BOARD_WIDTH):
            if state._grid[i][j] == color:
                if is_stable(state._grid, color, i, j):
                    # stables.append((i, j))  # Store the position of stable discs
                    n += 1
    # return stables
    return n


#
# the main body function:
#
def get_features(state: GameState, action: PointType) -> Counter:
    """
    The features of a current "game_state"
    (i.e. a position with the situation of the board)
    followed by an "action" to take.
    The color is neede to tell which side is mine to try to act optimally.

    Q_value = weights * features
    """

    #
    #
    ######## PREPARE ########
    #

    action_x, action_y = action
    opponent_state = state.get_successor(action)
    # the successor state (the next state) is the opponent's turn

    # the same as GameState.is_placeable in game_state.py
    color = BLACK if state.status == StateStatus.BLACK_TURN else WHITE
    opponent_color = -color

    # legal actions and counts of them
    legal_actions = state.legal_actions
    opponent_legal_actions = opponent_state.legal_actions
    n_actions = len(legal_actions)
    n_oppoent_actions = len(opponent_legal_actions)

    #
    #
    ######## FEATURES ########
    #

    features = Counter()  # all values are 0 defaultly

    features["bias"] = 1.0  # may adjust?

    # if the action is to put my chess on a corner, then that's great!
    if action in CORNERS:
        features["is_corner"] = 4.0  # may adjust?

    # threat that opponent would take the corner,
    # kind of overlap with CORNER_NEIGHBOURS
    for a in opponent_legal_actions:
        if a in CORNERS:
            features["corner_threat"] = -2  # may adjust?

    # if the action is to put my chess on a position next to corner,
    # then that's bad, which would probably make me lose the corner!
    if action in star_positions:
        if features["corner_threat"] != 0:
            features["is_corner_neighbour"] = -1.0  # may adjust?
    # the rest positions on edges are "subopitimal"
    elif (
        action_x == 0
        or action_x == BOARD_WIDTH - 1
        or action_y == 0
        or action_y == BOARD_WIDTH - 1
    ):
        features["is_edge"] = 1.0  # may adjust?

    # evaluate the changes that the action would make on my state
    features["flips"] = B_W_counts(opponent_state, color) - B_W_counts(state, color)
    # may adjust the feature:
    features["flips"] = features["flips"] / 20.0

    # Our control over the chess board:
    # opponent has action(s) to take
    if opponent_state.legal_actions:
        features["mobility_ratio"] = n_actions / n_oppoent_actions
        # may adjust the feature:
        features["mobility_ratio"] = features["mobility_ratio"] / 16.0
    # opponent has no action to take
    else:
        features["mobility_ratio"] = 1.0  # may adjust?

    # Addition od positions of my color that won't be changed
    features["stable_add"] = n_stables(opponent_state, color) - n_stables(state, color)
    # may adjust the feature:
    features["stable_add"] = features["stable_add"] / 16.0

    return features
