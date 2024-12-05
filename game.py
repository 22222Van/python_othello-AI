from agents import BaseAgent
from game_state import GameState, GameStatus

import ui

from utils import *


class Game():
    def __init__(
        self,
        black_agent: BaseAgent,
        white_agent: BaseAgent,
        use_graphic: bool = True
    ):
        super().__init__()

        self.black_agent = black_agent
        self.white_agent = white_agent
        self.use_graphic = use_graphic

        self.game_state = GameState()

        if use_graphic:
            ui.init_window()

    def start(self):
        while self.game_state.running:
            if self.use_graphic:
                self.game_state.draw_board()
            print(f'{self.game_state}\n')  # debug

            color: ColorType = (
                'B' if self.game_state.status == GameStatus.BLACK else 'W'
            )
            legal_actions = self.game_state.legal_actions

            if len(legal_actions) != 0:
                action = (
                    self.black_agent.get_action(self.game_state) if color == 'B'
                    else self.white_agent.get_action(self.game_state)
                )
                self.game_state = self.game_state.get_successor(action)
            else:
                # 空过的情况
                self.game_state = self.game_state.get_successor(None)

        # 游戏结束，结算
        black_count, white_count = self.game_state.black_white_counts

        print(f'{self.game_state}')
        print(f'Black {black_count}-{white_count} White')
