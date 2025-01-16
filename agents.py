import random
from game_state import GameState
from heuristics import BaseHeuristic

from abc import ABC, abstractmethod
from utils import *
import ui

import pickle


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    registry: Dict[str, Type['BaseAgent']] = {}

    def __init__(self, color: PlayerColorType) -> None:
        super().__init__()
        self.color = color

    def __init_subclass__(cls, cli_name: Optional[str] = None) -> None:
        super().__init_subclass__()

        if cli_name is None:
            cli_name = cls.__name__

        cls.registry[cli_name.lower()] = cls

    @abstractmethod
    def get_action(self, game_state: GameState) -> PointType:
        ...


class Player(BaseAgent):
    def get_action(self, game_state):
        return ui.get_player_action(game_state.legal_actions)


class RandomAgent(BaseAgent):
    def get_action(self, game_state):
        return random.choice(game_state.legal_actions)


class UpperLeftLover(BaseAgent):
    def get_action(self, game_state):
        return game_state.legal_actions[0]


class Limiter(BaseAgent):
    """
        An evil agent who uses greedy search to minimize the number of legal
        actions of the opponent.
    """
    # 这个 Agent 是用来 debug 没有 legal actions，但是游戏没有结束的情况的
    # 正好也可以作为一个示例

    def get_action(self, game_state):
        min_choices_number = 99999
        # game_state.legal_actions 保证一定非空（大概，否则就是 game.py 有 bug）
        select_action = game_state.legal_actions[0]
        for action in game_state.legal_actions:
            successor_state = game_state.get_successor(action)
            choices_number = len(successor_state.legal_actions)
            if choices_number < min_choices_number:
                select_action = action
                min_choices_number = choices_number

        return select_action


class InformedAgent(BaseAgent):
    """
        Abstract base class for all agents which have an heuristic function.
    """

    def __init__(self, color: PlayerColorType, heuristic: str) -> None:
        super().__init__(color)
        self.heuristic = BaseHeuristic.registry[heuristic.lower()](color)


class GreedyAgent(InformedAgent):
    """
        An evil agent who uses greedy search to maximize it's heuristic.
        If multiple actions result in the same maximum value, it will randomly
        select one of those actions.
    """

    def get_action(self, game_state):
        values = [
            self.heuristic(game_state.get_successor(a))
            for a in game_state.legal_actions
        ]
        max_value = max(values)
        max_action_list = [
            a
            for a, v in zip(game_state.legal_actions, values)
            if v == max_value
        ]
        return random.choice(max_action_list)


class MinimaxAgent(InformedAgent):
    def __init__(
        self,
        color: PlayerColorType,
        heuristic: str,
        depth: int,
        branch: Optional[int] = None
    ):
        super().__init__(color, heuristic)
        self.depth = depth
        self.branch = branch

    def get_action(self, game_state):
        def max_node(
            state: GameState, cur_depth: int, alpha: float, beta: float
        ) -> Tuple[float, Optional[PointType]]:
            if cur_depth >= self.depth:
                return self.heuristic(state), None
            if not state.running:
                return self.heuristic(state), None

            v = -INF
            legal_actions = state.legal_actions
            if not legal_actions:
                legal_actions = [None]

            legal_actions_and_successors = [
                (a, state.get_successor(a)) for a in legal_actions
            ]
            random.shuffle(legal_actions_and_successors)
            legal_actions_and_successors.sort(
                key=lambda x: self.heuristic(x[1]),
                reverse=True
            )

            if self.branch is not None:
                legal_actions_and_successors[self.branch:] = []

            max_action = None
            for action, successor in legal_actions_and_successors:
                cur_v = min_node(successor, cur_depth, alpha, beta)
                if v < cur_v:
                    v = cur_v
                    max_action = action
                if v >= beta:
                    break
                alpha = max(alpha, v)
            return v, max_action

        def min_node(
            state: GameState, cur_depth: int, alpha: float, beta: float
        ) -> float:
            assert cur_depth < self.depth
            if not state.running:
                return self.heuristic(state)

            v = INF
            legal_actions = state.legal_actions
            if not legal_actions:
                legal_actions = [None]

            legal_actions_and_successors = [
                (a, state.get_successor(a)) for a in legal_actions
            ]
            random.shuffle(legal_actions_and_successors)
            legal_actions_and_successors.sort(
                key=lambda x: self.heuristic(x[1]),
                reverse=False
            )

            if self.branch is not None:
                legal_actions_and_successors[self.branch:] = []

            for _, successor in legal_actions_and_successors:
                cur_v, _ = max_node(successor, cur_depth+1, alpha, beta)
                v = min(v, cur_v)
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        _, max_action = max_node(game_state, 0, -INF, INF)

        assert max_action is not None
        return max_action


class DeepLearningAgent(BaseAgent):
    path_table = {
        'cnn_arch_1': './saves/cnn_arch_1.pth',
        'cnn_arch_2': './saves/cnn_arch_2.pth',
        'cnn_arch_3': './saves/cnn_arch_3.pth',
        'cnn_arch_4': './saves/cnn_arch_4.pth',
        'transformer': './saves/transformer.pth',
    }

    def __init__(
        self,
        color: PlayerColorType,
        model: str = 'cnn_arch_2',
    ):
        from torch_models import (
            cnn_arch_1,
            cnn_arch_2,
            cnn_arch_3,
            cnn_arch_4,
            mlp,
            transformer,
            predict
        )
        import torch
        if TYPE_CHECKING:
            from torch import nn

        super().__init__(color)

        if model == 'cnn_arch_1':
            self.model: nn.Module = cnn_arch_1()
        elif model == 'cnn_arch_2':
            self.model: nn.Module = cnn_arch_2()
        elif model == 'cnn_arch_3':
            self.model: nn.Module = cnn_arch_3()
        elif model == 'cnn_arch_4':
            self.model: nn.Module = cnn_arch_4()
        elif model == 'transformer':
            self.model: nn.Module = transformer()
        else:
            raise ValueError(f"Unknown model `{model}`")
        self.model.load_state_dict(
            torch.load(
                self.path_table[model], map_location=torch.device('cpu')
            )
        )
        self.model.eval()
        self.predict = predict

    def get_action(self, game_state):
        board = game_state.grid
        player = 1 if self.color == BLACK else 0
        legal_list = game_state.legal_actions
        action = self.predict(self.model, board, player, legal_list)
        return action


class ApproximateQAgent(BaseAgent):
    def __init__(self, color: PlayerColorType, extractor='IdentityExtractor', alpha=0.01, discount=0.9):
        """
        Initializes an ApproximateQAgent.

        Args:
            color (PlayerColorType): The color of the agent.
            extractor (str): The name of the feature extractor to use.
            alpha (float): The learning rate.
            discount (float): The discount factor.
        """
        super().__init__(color)
        self.featExtractor = globals()[extractor]()
        self.weights = {}
        self.alpha = alpha
        self.discount = discount

    def getWeights(self):
        """
        Returns the current weights of the agent.
        """
        return self.weights

    def getQValue(self, state, action):
        """
        Returns the Q-value for the given state and action.

        Args:
            state (GameState): The current state of the game.
            action (PointType): The action to take.

        Returns:
            float: The Q-value.
        """
        ans = 0
        features = self.featExtractor.getFeatures(state, action)
        for f in features:
            ans += features[f] * self.weights.get(f, 0.0)
        return ans

    def save_weights(self, filepath: str) -> None:
        """
        Saves the agent's weights to a file using pickle.

        Args:
            filepath (str): The path to the file where the weights will be saved.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self.weights, file)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath: str) -> None:
        """
        Loads the agent's weights from a file using pickle.

        Args:
            filepath (str): The path to the file from which the weights will be loaded.
        """
        try:
            with open(filepath, 'rb') as file:
                self.weights = pickle.load(file)
            print(f"Weights loaded from {filepath}")
        except FileNotFoundError:
            print(
                f"No weights file found at {filepath}. Starting with empty weights.")
        except Exception as e:
            print(f"An error occurred while loading weights: {e}")

    def update(self, state, action, nextState, reward: float):
        """
        Updates the weights based on the given experience.

        Args:
            state (GameState): The current state of the game.
            action (PointType): The action taken.
            nextState (GameState): The next state of the game.
            reward (float): The reward received.
        """
        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * (0 if len(nextState.legal_actions) == 0 else max(
            self.getQValue(nextState, a)for a in nextState.legal_actions))) - self.getQValue(state, action)
        for f in features:
            self.weights[f] = self.weights.get(
                f, 0.0) + self.alpha * difference * features[f]

    def final(self, state):
        """
        Called at the end of the game.
        """
        pass
