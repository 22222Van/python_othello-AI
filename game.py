from enum import Enum
from agents import BaseAgent
import ui

from utils import *


class GameStatus(Enum):
    BLACK = 1
    WHITE = 2
    # 下面三个目前代码里永远不会出现，但是之后可能会有用
    BLACK_WIN = 3
    WHITE_WIN = 4
    DRAW = 5


class GameState():
    def __init__(self):
        super().__init__()

        str_board = [
            ['|' for _ in range(BOARD_WIDTH)]
            for _ in range(BOARD_WIDTH)
        ]
        str_board[3][3], str_board[4][4] = 'W', 'W'
        str_board[3][4], str_board[4][3] = 'B', 'B'
        self.grid: GridType = str_board

        self.status: GameStatus = GameStatus.BLACK

    def print_grid(self, legal_actions) -> None:
        disp_mat = [[' ' for _ in range(BOARD_WIDTH+1)]
                    for _ in range(BOARD_WIDTH+1)]
        for i in range(1, BOARD_WIDTH+1):
            disp_mat[0][i] = str(i-1)
            disp_mat[i][0] = str(i-1)
        for j in range(1, BOARD_WIDTH+1):
            for k in range(1, BOARD_WIDTH+1):
                if (j-1, k-1) in legal_actions:
                    disp_mat[j][k] = '*'
                else:
                    disp_mat[j][k] = self.grid[j-1][k-1]
        for row in disp_mat:
            print(' '.join(row))

    def draw(self, legal_actions) -> None:
        color = 'B' if self.status == GameStatus.BLACK else 'W'
        ui.draw_board(self.grid, color, legal_actions)

    def running(self) -> bool:
        return self.status == GameStatus.BLACK or self.status == GameStatus.WHITE

    def update(self, action: Optional[PointType], color: ColorType) -> None:
        # print(action, color)
        if action is not None:
            x, y = action
            self.grid[x][y] = color
            for i in range(1, 9):
                self.flip(i, (x, y), color)

        if color == 'B':
            self.status = GameStatus.WHITE
        else:
            self.status = GameStatus.BLACK

    def flip(self, direction, position, color):
        """ Flips (capturates) the pieces of the given color in the given direction
        (1=North,2=Northeast...) from position. """
        if direction == 1:
            # north
            row_inc = -1
            col_inc = 0
        elif direction == 2:
            # northeast
            row_inc = -1
            col_inc = 1
        elif direction == 3:
            # east
            row_inc = 0
            col_inc = 1
        elif direction == 4:
            # southeast
            row_inc = 1
            col_inc = 1
        elif direction == 5:
            # south
            row_inc = 1
            col_inc = 0
        elif direction == 6:
            # southwest
            row_inc = 1
            col_inc = -1
        elif direction == 7:
            # west
            row_inc = 0
            col_inc = -1
        elif direction == 8:
            # northwest
            row_inc = -1
            col_inc = -1
        else:
            row_inc, col_inc = 0, 0
            raise ValueError("Illegal direction occured")

        places = []     # pieces to flip
        i = position[0] + row_inc
        j = position[1] + col_inc

        other = 'W'
        if color == 'W':
            other = 'B'

        if i in range(8) and j in range(8) and self.grid[i][j] == other:
            # assures there is at least one piece to flip
            places = places + [(i, j)]
            i = i + row_inc
            j = j + col_inc
            while i in range(8) and j in range(8) and self.grid[i][j] == other:
                # search for more pieces to flip
                places = places + [(i, j)]
                i = i + row_inc
                j = j + col_inc
            if i in range(8) and j in range(8) and self.grid[i][j] == color:
                # found a piece of the right color to flip the pieces between
                for pos in places:
                    # flips
                    self.grid[pos[0]][pos[1]] = color

    def is_slot_legal(self, x: int, y: int) -> bool:
        """
        Check whether ((0<=x<BOARD_WIDTH) and (0<=y<BOARD_WIDTH))
        """
        board_width = len(self.grid)
        return (x >= 0) and (x < board_width) and (y >= 0) and (y < board_width)

    def is_cell_placeable(self, x: int, y: int, color: ColorType) -> List[PointType]:
        """
        Get the placeable neighbours of board[x][y]
        """
        opponent_color = 'W'
        if color == 'W':
            opponent_color = 'B'

        output = []
        if self.grid[x][y] != opponent_color:
            return output

        for i in [1, 0, -1]:
            for j in [1, 0, -1]:
                if not ((i == 0) and (j == 0)) and self.is_slot_legal(x+i, y+j):
                    legal = False
                    if self.grid[x+i][y+j] == '|':
                        temp_x = x-i
                        temp_y = y-j
                        while self.is_slot_legal(temp_x, temp_y):
                            if self.grid[temp_x][temp_y] == color:
                                legal = True
                                break
                            if self.grid[temp_x][temp_y] != opponent_color:
                                break
                            temp_y -= j
                            temp_x -= i
                    if legal:
                        output.append((x+i, y+j))
        return output

    def get_legal_actions(self, color: ColorType) -> List[PointType]:
        placeable = []
        for i in range(BOARD_WIDTH):
            for j in range(BOARD_WIDTH):
                i_j_placeable = self.is_cell_placeable(i, j, color)
                for x, y in i_j_placeable:
                    placeable.append((x, y))
        return placeable


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
        no_legal_actions_flag = False
        while self.game_state.running():
            color: ColorType = (
                'B' if self.game_state.status == GameStatus.BLACK else 'W'
            )
            legal_actions = self.game_state.get_legal_actions(color)
            if self.use_graphic:
                self.game_state.draw(legal_actions)
            if legal_actions:
                no_legal_actions_flag = False
                action = (
                    self.black_agent.get_action(legal_actions) if color == 'B'
                    else self.white_agent.get_action(legal_actions)
                )
                self.game_state.update(action, color)
            else:
                if no_legal_actions_flag:
                    break
                else:
                    self.game_state.update(None, color)
                    no_legal_actions_flag = True


        # 游戏结束，结算
        black_count = 0
        white_count = 0
        for i in range(BOARD_WIDTH):
            for j in range(BOARD_WIDTH):
                if self.game_state.grid[i][j]=='B':
                    black_count+=1
                if self.game_state.grid[i][j]=='W':
                    white_count+=1

        result_str=""
        if (black_count>white_count):
            result_str="Black Wins!"
        elif (black_count==white_count):
            result_str="Tie!"
        else:
            result_str="White Wins!"
        print(result_str)
        print(f'Black {black_count}-{white_count} White')
