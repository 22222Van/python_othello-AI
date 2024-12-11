import random
from game_state import GameState
from game_state import GameStatus

from abc import ABC, abstractmethod
from utils import *
import ui


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    registry: dict[str, Type['BaseAgent']] = {}

    def __init__(self, color: ColorType) -> None:
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


class GreedyAgent(BaseAgent):
    """
        An evil agent who uses greedy search to maximize the number of
        it's color's pieces.
        When more than one action lead to the maximization,
        randomly choose one.
    """

    def get_action(self, game_state):
        count_act_list = [(game_state.get_successor(a).black_white_counts, a)
                          for a in game_state.legal_actions]
        max_num = -1
        max_action_list = []
        if game_state.status == GameStatus.BLACK:
            for count_act in count_act_list:
                if (count_act[0][0] > max_num):
                    max_action_list = [count_act[1]]
                    max_num = count_act[0][0]
                elif (count_act[0][0] == max_num):
                    max_action_list.append(count_act[1])
        else:
            for count_act in count_act_list:
                if (count_act[0][1] > max_num):
                    max_action_list = [count_act[1]]
                    max_num = count_act[0][1]
                elif (count_act[0][1] == max_num):
                    max_action_list.append(count_act[1])
        return random.choice(max_action_list)


class MinimaxAgent(BaseAgent):  # FIXME: Still have bug when depth=2
    def __init__(self, color: ColorType, depth: int):
        super().__init__(color)
        self.depth = depth

    def get_action(self, game_state):
        superior_action = None
        max_value = None
        actionList = game_state.legal_actions

        for action in actionList:

            temp = self.naive_evaluation_func(game_state, game_state.status, 0)

            if max_value == None and superior_action == None:
                max_value = temp
                superior_action = action
            if temp > max_value:
                max_value = temp
                superior_action = action

        assert superior_action is not None

        return superior_action

    def naive_evaluation_func(self, game_state: GameState, color, depth):
        """
        Black's Naive evaluation that just counts the number of pieces directly
        """

        if (len(game_state.legal_actions) == 0):
            return (game_state.black_white_counts)[0] if color == GameStatus.BLACK else (game_state.black_white_counts)[1]

        if game_state.running == False:
            return (game_state.black_white_counts)[0] if color == GameStatus.BLACK else (game_state.black_white_counts)[1]

        evalutaion = 0

        if color == GameStatus.BLACK:
            if game_state.status == GameStatus.BLACK:
                new_depth = depth+1
                if new_depth == self.depth:
                    return (game_state.black_white_counts)[0]
                evalutaion = self.max_B_value(game_state, color, depth)
            else:
                evalutaion = self.min_B_value(game_state, color, depth)
        else:
            if game_state.status == GameStatus.WHITE:
                new_depth = depth+1
                if new_depth == self.depth:
                    return (game_state.black_white_counts)[1]
                evalutaion = self.max_W_value(game_state, color, depth)
            else:
                evalutaion = self.min_W_value(game_state, color, depth)
        return evalutaion

    def min_B_value(self, game_state: GameState, color, depth):
        """
        returns the minimum value of Black Pieces
        """
        value = None
        actionList = game_state.legal_actions
        if game_state.running == False:
            return (game_state.black_white_counts)[0]
        if len(actionList) == 0:
            return self.naive_evaluation_func(game_state.get_successor(None), color, depth)

        for action in actionList:
            temp = self.naive_evaluation_func(
                game_state.get_successor(action), color, depth)
            if value == None:
                value = temp
            if temp < value:
                value = temp

        return value

    def max_B_value(self, game_state: GameState, color, depth):
        """
        returns the minimum value of Black Pieces
        """
        value = None
        actionList = game_state.legal_actions
        if game_state.running == False:
            return (game_state.black_white_counts)[0]
        if len(actionList) == 0:
            return self.naive_evaluation_func(game_state.get_successor(None), color, depth)

        for action in actionList:
            temp = self.naive_evaluation_func(
                game_state.get_successor(action), color, depth)
            if value == None:
                value = temp
            if temp > value:
                value = temp
        return value

    # def naive_W_evaluation_func(self, game_state:GameState, depth):
    #     """
    #     White's Naive evaluation that just counts the number of pieces directly
    #     """
    #     if (len(game_state.legal_actions) == 0):
    #         return (game_state.black_white_counts())[1]

    #     if game_state.running == False:
    #         return (game_state.black_white_counts())[1]

    #     evalutaion = 0

    #     if game_state.status() == GameStatus.WHITE:
    #         new_depth = depth+1
    #         if new_depth == self.depth:
    #             return (self.black_white_counts())[1]
    #         evalutaion=self.max_W_value(game_state,depth)
    #     else:
    #         evalutaion = self.min_W_value(game_state, depth)
    #     return evalutaion

    def min_W_value(self, game_state: GameState, color, depth):
        """
        returns the minimum value of White Pieces
        """
        value = None
        actionList = game_state.legal_actions
        if game_state.running == False:
            return (game_state.black_white_counts)[1]
        if len(actionList) == 0:
            return self.naive_evaluation_func(game_state.get_successor(None), color, depth)

        for action in actionList:
            temp = self.naive_evaluation_func(
                game_state.get_successor(action), color, depth)
            if value == None:
                value = temp
            if temp < value:
                value = temp

        return value

    def max_W_value(self, game_state: GameState, color, depth):
        """
        returns the minimum value of White Pieces
        """
        value = None
        actionList = game_state.legal_actions
        if game_state.running == False:
            return (game_state.black_white_counts)[1]
        if len(actionList) == 0:
            return self.naive_evaluation_func(game_state.get_successor(None), color, depth)

        for action in actionList:
            temp = self.naive_evaluation_func(
                game_state.get_successor(action), color, depth)
            if value == None:
                value = temp
            if temp > value:
                value = temp
        return value
