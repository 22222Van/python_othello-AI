from game_state import GameState
from game_state import StateStatus

from abc import ABC, abstractmethod
from collections import defaultdict
from utils import *


class BaseExtractor(ABC):
    """
    Abstract base class for all evaluation functions.
    """
    registry: Dict[str, Type['BaseExtractor']] = {}

    def __init__(self, color: PlayerColorType) -> None:
        super().__init__()
        self.color = color

    def __init_subclass__(cls, cli_name: Optional[str] = None) -> None:
        super().__init_subclass__()

        if cli_name is None:
            cli_name = cls.__name__

        cls.registry[cli_name.lower()] = cls

    def __call__(
        self, game_state: GameState, action: PointType
    ) -> Dict[Any, float]:
        return self.get_features(game_state, action)

    @abstractmethod
    def get_features(
        self, state: GameState, action: PointType
    ) -> Dict[Any, float]:
        ...


class IdentityExtractor(BaseExtractor):
    def get_features(self, state, action):
        feats = defaultdict(float)
        feats[(state, action)] = 1.0
        return feats


class NaiveExtractor(BaseExtractor):
    """
        Assume that the two pieces on the chessboard are independent.
        The meaning of "Naive" is the same as in "Naive" Bayes.
    """

    norm = BOARD_WIDTH * BOARD_WIDTH

    def get_features(self, state, action):
        feats = defaultdict(float)
        next_state = state.get_successor(action)

        factor = self.color / self.norm

        feats['bias'] = 1 / self.norm

        for i in range(BOARD_WIDTH):
            for j in range(BOARD_WIDTH):
                feats[(i, j)] = next_state.grid[i, j] * factor
        return feats


class MixedExtractor(BaseExtractor):
    CORNERS = [
        (0, 0),
        (BOARD_WIDTH - 1, 0),
        (0, BOARD_WIDTH - 1),
        (BOARD_WIDTH - 1, BOARD_WIDTH - 1),
    ]

    def B_W_counts(self, game_state: GameState, color: PlayerColorType) -> float:
        bn, wn = game_state.black_white_counts
        if color == BLACK:
            return bn  # count of black
        else:
            return wn  # count of white

    def get_features(self, state, action):
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

        features = defaultdict(float)  # all values are 0 defaultly

        features["bias"] = 1.0

        # if the action is to put my chess on a corner, then that's great!
        if action in self.CORNERS:
            features["is_corner"] = 4.0

        # threat that opponent would take the corner,
        # kind of overlap with CORNER_NEIGHBOURS
        for a in opponent_legal_actions:
            if a in self.CORNERS:
                features["corner_threat"] = -2

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
        features["flips"] = self.B_W_counts(
            opponent_state, color) - self.B_W_counts(state, color)
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
