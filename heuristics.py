from game_state import GameState

from abc import ABC, abstractmethod
from utils import *


class BaseHeuristic(ABC):
    """
    Abstract base class for all evaluation functions.
    """
    registry: dict[str, Type['BaseHeuristic']] = {}

    def __init__(self, color: ColorType) -> None:
        super().__init__()
        self.color: ColorType = color

    def __init_subclass__(cls, cli_name: Optional[str] = None) -> None:
        super().__init_subclass__()

        if cli_name is None:
            cli_name = cls.__name__

        cls.registry[cli_name.lower()] = cls

    def __call__(
        self, game_state: GameState, color: Optional[ColorType] = None
    ) -> float:
        if color is None:
            color = self.color
        return self.evaluate(game_state, color)

    @abstractmethod
    def evaluate(self, game_state: GameState, color: ColorType) -> float:
        ...


class CountOwnPieces(BaseHeuristic):
    def evaluate(self, game_state, color) -> float:
        bn, wn = game_state.black_white_counts
        if color == 'B':
            return bn
        else:
            return wn


class CountPiecesDifference(BaseHeuristic):
    def evaluate(self, game_state, color) -> float:
        bn, wn = game_state.black_white_counts
        v = bn-wn
        if color == 'B':
            return v
        else:
            return -v
