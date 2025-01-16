from utils import *
from game_state import StateStatus, GameState


# the same as CountOwnPieces.evaluate() in heuristics.py
def B_W_counts(game_state: GameState, color: PlayerColorType) -> float:
    bn, wn = game_state.black_white_counts
    if color == BLACK:
        return bn  # count of black
    else:
        return wn  # count of white


# stable pieces (i.e. position with chess) that will not be flipped
def is_stable():
    1  # TBD


def num_stables():
    2  # TBD


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

    features["bias"] = 1.0

    # if the action is to put my chess on a corner, then that's great!
    if action in CORNERS:
        features["is_corner"] = 4.0

    # threat that opponent would take the corner,
    # kind of overlap with CORNER_NEIGHBOURS
    for a in opponent_legal_actions:
        if a in CORNERS:
            features["corner_threat"] -= 2

    # if the action is to put my chess on a position next to corner,
    # then that's bad, which would probably make me lose the corner!
    if action in star_positions:
        if features["corner_threat"] != 0:
            features["is_corner_neighbour"] = -1.0
    # the rest positions on edges are "subopitimal"
    elif (
        action_x == 0
        or action_x == BOARD_WIDTH - 1
        or action_y == 0
        or action_y == BOARD_WIDTH - 1
    ):
        features["is_edge"] = 1.0

    # evaluate the changes that the action would make on my state
    features["flips"] = B_W_counts(opponent_state, color) - B_W_counts(state, color)
    # may adjust the feature:
    features["flips"] = features["flips"] / 16.0

    # opponent has action(s) to take
    if opponent_state.legal_actions:
        features["mobility_ratio"] = len(state.legal_actions) / len(
            opponent_state.legal_actions
        )
        # may adjust the feature:
        features["mobility_ratio"] = features["mobility_ratio"] / 16.0
    # opponent has no action to take
    else:
        features["mobility_ratio"] = 1.0

    return features
