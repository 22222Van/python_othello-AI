from agents import BaseAgent, ApproximateQAgent
from game_state import GameState, StateStatus

import ui

from utils import *


class Game():
    def __init__(
        self,
        black_agent: BaseAgent,
        white_agent: BaseAgent,
        use_graphic: bool = True,
        debug: bool = False
    ):
        super().__init__()

        self.black_agent = black_agent
        self.white_agent = white_agent
        self.use_graphic = use_graphic
        self.debug = debug

        self.game_state = GameState()

        if use_graphic:
            ui.init_window()

    def start(self, train=False) -> GameState:
        black_train = False
        white_train = False

        if train:
            if isinstance(self.black_agent, ApproximateQAgent):
                black_train = True
            if isinstance(self.white_agent, ApproximateQAgent):
                white_train = True

        while self.game_state.running:
            if self.use_graphic:
                self.game_state.draw_board()
            if self.debug:
                print(f'\n{self.game_state}')  # debug

            color: PlayerColorType = (
                BLACK
                if self.game_state.status == StateStatus.BLACK_TURN
                else WHITE
            )
            legal_actions = self.game_state.legal_actions

            if len(legal_actions) != 0:
                action = (
                    self.black_agent.get_action(self.game_state)
                    if color == BLACK
                    else self.white_agent.get_action(self.game_state)
                )
                if self.debug:
                    print(action)

                if train:  # 如果要训练RL，缓存当前的动作
                    last_state = self.game_state

                self.game_state = self.game_state.get_successor(action)

                if train:  # 如果要训练RL，给reward
                    status = self.game_state.status
                    if status == StateStatus.BLACK_WINS:
                        reward = BLACK
                    elif status == StateStatus.WHITE_WINS:
                        reward = WHITE
                    else:
                        reward = EMPTY

                    if black_train:
                        self.black_agent.update(
                            last_state,  # type: ignore
                            action,
                            self.game_state,
                            reward
                        )
                    if white_train:
                        self.white_agent.update(
                            last_state,  # type: ignore
                            action,
                            self.game_state,
                            -reward
                        )

            else:
                # 空过的情况
                if self.debug:
                    print("No move, pass")
                self.game_state = self.game_state.get_successor(None)

            if self.debug:
                input("DEBUG: Press <Enter> / <Return> to continue...")

        if self.debug:
            print(f'{self.game_state}')


        return self.game_state
