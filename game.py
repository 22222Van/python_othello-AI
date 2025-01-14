from agents import BaseAgent
from game_state import GameState, GameStatus

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

    def start(self) -> GameState:
        while self.game_state.running:
            if self.use_graphic:
                self.game_state.draw_board()
            if self.debug:
                print(f'\n{self.game_state}')  # debug

            color: PlayerColorType = (
                BLACK if self.game_state.status == GameStatus.BLACK else WHITE
            )
            legal_actions = self.game_state.legal_actions

            if len(legal_actions) != 0:
                action = (
                    self.black_agent.get_action(self.game_state) if color == BLACK
                    else self.white_agent.get_action(self.game_state)
                )
                if self.debug:
                    print(action)
                self.game_state = self.game_state.get_successor(action)
            else:
                # 空过的情况
                if self.debug:
                    print("No move, pass")
                self.game_state = self.game_state.get_successor(None)

        if self.debug:
            print(f'{self.game_state}')

        return self.game_state
