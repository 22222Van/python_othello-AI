from enum import Enum
import copy
import ui
from utils import *


class GameStatus(Enum):
    BLACK = 1
    WHITE = 2
    BLACK_WIN = 3
    WHITE_WIN = 4
    DRAW = 5


class GameState():
    def __init__(self, other: Optional['GameState'] = None):
        """
            Copy construct if `other` is another GameState object.
        """
        super().__init__()

        if other is None:
            str_board = [
                ['|' for _ in range(BOARD_WIDTH)]
                for _ in range(BOARD_WIDTH)
            ]
            str_board[3][3], str_board[4][4] = 'W', 'W'
            str_board[3][4], str_board[4][3] = 'B', 'B'
            self._grid: GridType = str_board
            self._status: GameStatus = GameStatus.BLACK
        else:
            self._grid = copy.deepcopy(other._grid)
            self._status = copy.deepcopy(other._status)

        # deepcopy的时候不能拷贝cache！
        self.__successors_cache = {}

    # @property起保护作用，能阻止对grid，status赋值的行为

    @property
    def grid(self) -> GridType:
        return self._grid

    @property
    def status(self) -> GameStatus:
        return self._status

    # 复制GameState相关

    def __deepcopy__(self, memo: dict[int, Any]) -> 'GameState':
        if id(self) in memo:
            return memo[id(self)]
        return GameState(self)

    def clone(self) -> 'GameState':
        return self.__deepcopy__({})

    # 棋局数据相关

    @property
    def running(self) -> bool:
        return (
            self.status == GameStatus.BLACK or self.status == GameStatus.WHITE
        )

    @lazy_property
    def black_white_counts(self) -> Tuple[int, int]:
        black_count = 0
        white_count = 0
        for i in range(BOARD_WIDTH):
            for j in range(BOARD_WIDTH):
                if self.grid[i][j] == 'B':
                    black_count += 1
                if self.grid[i][j] == 'W':
                    white_count += 1
        return black_count, white_count

    # 展示棋局相关

    def __str__(self) -> str:
        return_str = ""

        disp_mat = [[' ' for _ in range(BOARD_WIDTH+1)]
                    for _ in range(BOARD_WIDTH+1)]
        for i in range(1, BOARD_WIDTH+1):
            disp_mat[0][i] = str(i-1)
            disp_mat[i][0] = str(i-1)
        for j in range(1, BOARD_WIDTH+1):
            for k in range(1, BOARD_WIDTH+1):
                if (j-1, k-1) in self.legal_actions:
                    disp_mat[j][k] = '*'
                else:
                    disp_mat[j][k] = self.grid[j-1][k-1]
        for row in disp_mat:
            return_str += ' '.join(row)
            return_str += '\n'

        if self.status == GameStatus.BLACK:
            return_str += 'Turn: Black'
        elif self.status == GameStatus.WHITE:
            return_str += 'Turn: White'
        elif self.status == GameStatus.BLACK_WIN:
            return_str += 'Black Wins!'
        elif self.status == GameStatus.WHITE_WIN:
            return_str += 'White Wins!'
        elif self.status == GameStatus.DRAW:
            return_str += 'Draw!'

        return return_str

    def draw_board(self) -> None:
        color = 'B' if self.status == GameStatus.BLACK else 'W'
        ui.draw_board(self.grid, color, self.legal_actions)

    # legal_actions、get_successor、翻转棋子算法相关

    @staticmethod
    def is_in_board(x: int, y: int) -> bool:
        """
        Check whether `0<=x<BOARD_WIDTH and 0<=y<BOARD_WIDTH`
        """
        return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_WIDTH

    def is_placeable(self, x: int, y: int) -> List[PointType]:
        """
        Get the placeable neighbours of board[x][y]
        """
        color = 'B' if self.status == GameStatus.BLACK else 'W'

        opponent_color = 'W'
        if color == 'W':
            opponent_color = 'B'

        output = []
        if self._grid[x][y] != opponent_color:
            return output

        for i in [1, 0, -1]:
            for j in [1, 0, -1]:
                if not ((i == 0) and (j == 0)) and self.is_in_board(x+i, y+j):
                    legal = False
                    if self._grid[x+i][y+j] == '|':
                        temp_x = x-i
                        temp_y = y-j
                        while self.is_in_board(temp_x, temp_y):
                            if self._grid[temp_x][temp_y] == color:
                                legal = True
                                break
                            if self._grid[temp_x][temp_y] != opponent_color:
                                break
                            temp_y -= j
                            temp_x -= i
                    if legal:
                        output.append((x+i, y+j))
        return output

    def _flip(self, direction, position, color):
        """
            Flips (capturates) the pieces of the given color in the given
            direction (1=North,2=Northeast...) from position.
        """
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

        if i in range(8) and j in range(8) and self._grid[i][j] == other:
            # assures there is at least one piece to flip
            places = places + [(i, j)]
            i = i + row_inc
            j = j + col_inc
            while i in range(8) and j in range(8) and self._grid[i][j] == other:
                # search for more pieces to flip
                places = places + [(i, j)]
                i = i + row_inc
                j = j + col_inc
            if i in range(8) and j in range(8) and self._grid[i][j] == color:
                # found a piece of the right color to flip the pieces between
                for pos in places:
                    # flips
                    self._grid[pos[0]][pos[1]] = color

    def get_successor(self, action: Optional[PointType]) -> 'GameState':
        '''
            When `action` is `None`, the obtained successor will only swap the
            current player without making any changes to the board.
        '''
        def get_successor_helper() -> 'GameState':
            successor = self.clone()
            color = 'B' if self.status == GameStatus.BLACK else 'W'

            if action is not None:
                x, y = action
                successor._grid[x][y] = color
                for i in range(1, 9):
                    successor._flip(i, (x, y), color)

            # 不管 successor 的棋局是否已经结束，因为要获得 successor.legal_actions，
            # 所以必须暂时先给它赋一个 WHITE 或 BLACK 的值
            if color == 'B':
                successor._status = GameStatus.WHITE
            else:
                successor._status = GameStatus.BLACK

            if len(successor.legal_actions) == 0:
                grand_successor = successor.clone()
                grand_successor._status = self.status
                if len(grand_successor.legal_actions) == 0:
                    # 连续两个都没有 legal_actions，游戏已经结束
                    black_counts, white_counts = successor.black_white_counts
                    if black_counts > white_counts:
                        successor._status = GameStatus.BLACK_WIN
                    elif black_counts < white_counts:
                        successor._status = GameStatus.WHITE_WIN
                    else:
                        successor._status = GameStatus.DRAW
                    return successor
                else:
                    successor.__successors_cache[None] = grand_successor

            return successor

        if action not in self.__successors_cache:
            self.__successors_cache[action] = get_successor_helper()

        return self.__successors_cache[action]

    @lazy_property
    def legal_actions(self) -> List[PointType]:
        placeable = []
        for i in range(BOARD_WIDTH):
            for j in range(BOARD_WIDTH):
                i_j_placeable = self.is_placeable(i, j)
                for x, y in i_j_placeable:
                    if (x, y) not in placeable:
                        placeable.append((x, y))
        return placeable

    def B_naive_evaluation_func(self):
        """
        Black's Naive evaluation that just counts the number of pieces directly
        """

    def W_naive_evaluation_func(self):
        """
        White's Naive evaluation that just counts the number of pieces directly
        """
