import random

from abc import ABC, abstractmethod
from utils import *
import ui


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    registry: dict[str, Type['BaseAgent']] = {}

    def __init_subclass__(cls, cli_name: Optional[str] = None) -> None:
        super().__init_subclass__()

        if cli_name is None:
            cli_name = cls.__name__

        cls.registry[cli_name.lower()] = cls

    @abstractmethod
    def get_action(self, legal_actions: List[PointType]) -> PointType:
        ...


class Player(BaseAgent):
    def get_action(self, legal_actions):
        return ui.get_player_action(legal_actions)


# class FoolishAgent(BaseAgent):
#     def get_action(self, legal_actions):
#         assert legal_actions
#         return legal_actions[0]


class RandomAgent(BaseAgent):
    def get_action(self, legal_actions):
        assert legal_actions
        # legal_actions 可以变成一个 set 以加快访存速度
        legal_actions = list(set(legal_actions))
        return random.choice(legal_actions)

# class 