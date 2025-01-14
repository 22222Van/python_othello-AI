from game_state import GameState

from abc import ABC, abstractmethod
from utils import *


class BaseHeuristic(ABC):
    """
    Abstract base class for all evaluation functions.
    """
    registry: dict[str, Type['BaseHeuristic']] = {}

    def __init__(self, color: PlayerColorType) -> None:
        super().__init__()
        self.color: PlayerColorType = color

    def __init_subclass__(cls, cli_name: Optional[str] = None) -> None:
        super().__init_subclass__()

        if cli_name is None:
            cli_name = cls.__name__

        cls.registry[cli_name.lower()] = cls

    def __call__(
        self, game_state: GameState, color: Optional[PlayerColorType] = None
    ) -> float:
        if color is None:
            color = self.color
        return self.evaluate(game_state, color)

    @abstractmethod
    def evaluate(self, game_state: GameState, color: PlayerColorType) -> float:
        ...


class CountOwnPieces(BaseHeuristic):
    def evaluate(self, game_state, color) -> float:
        bn, wn = game_state.black_white_counts
        if color == BLACK:
            return bn
        else:
            return wn


class CountPiecesDifference(BaseHeuristic):
    def evaluate(self, game_state, color) -> float:
        bn, wn = game_state.black_white_counts
        v = bn-wn
        if color == BLACK:
            return v
        else:
            return -v

# FIXME:This may cause bugs in minimax
# class CountLegalActions(BaseHeuristic):
#     def evaluate(self, game_state, color) -> float:
#         return len(game_state.legal_actions)


class WeightedCountPieces(BaseHeuristic):
    """
    在原有每一个棋子得一分的基础上，
    棋子在角落额外得两分，棋子在边上额外得一分
    """
    def evaluate(self, game_state, color) -> float:
        black_score = 0
        white_score = 0

        bn, wn = game_state.black_white_counts
        black_score += bn
        white_score += wn

        #在原有每一个棋子得一分的基础上，
        #棋子在角落额外得两分，棋子在边上额外得一分
        for i in range(BOARD_WIDTH):
            if game_state.grid[0][i] == BLACK:
                black_score += 1
            if game_state.grid[0][i] == WHITE:
                white_score += 1
            if game_state.grid[BOARD_WIDTH-1][i] == BLACK:
                black_score += 1
            if game_state.grid[BOARD_WIDTH-1][i] == WHITE:
                white_score += 1
            
            if game_state.grid[i][0] == BLACK:
                black_score += 1
            if game_state.grid[i][0] == WHITE:
                white_score += 1
            if game_state.grid[i][BOARD_WIDTH-1] == BLACK:
                black_score += 1
            if game_state.grid[i][BOARD_WIDTH-1] == WHITE:
                white_score += 1

        if color == BLACK:
            return black_score
        else:
            return white_score

# class MixedHeuristic(BaseHeuristic):
#     def evaluate(self, game_state, color) -> float:
#         black_score = 0
#         white_score = 0

#         bn, wn = game_state.black_white_counts
#         black_score += bn
#         white_score += wn

#         #在原有每一个棋子得一分的基础上，
#         #棋子在角落额外得两分，棋子在边上额外得一分
#         for i in range(BOARD_WIDTH):
#             if game_state.grid[0][i] == BLACK:
#                 black_score += 1
#             if game_state.grid[0][i] == WHITE:
#                 white_score += 1
#             if game_state.grid[BOARD_WIDTH-1][i] == BLACK:
#                 black_score += 1
#             if game_state.grid[BOARD_WIDTH-1][i] == WHITE:
#                 white_score += 1
            
#             if game_state.grid[i][0] == BLACK:
#                 black_score += 1
#             if game_state.grid[i][0] == WHITE:
#                 white_score += 1
#             if game_state.grid[i][BOARD_WIDTH-1] == BLACK:
#                 black_score += 1
#             if game_state.grid[i][BOARD_WIDTH-1] == WHITE:
#                 white_score += 1

        # if color == BLACK:
        #     return black_score+len(game_state.legal_actions)
        # else:
        #     return white_score+len(game_state.legal_actions)

class MajorityOfRows(BaseHeuristic):
    """
    if a row has # of Black># of White, black_score++
    if a row has # of Black<# of White, white_score++
    """
    def evaluate(self, game_state, color) -> float:
        black_score=0
        white_score=0
        for i in range(BOARD_WIDTH):
            black_count=0
            white_count=0
            for j in range(BOARD_WIDTH):
                if game_state.grid[i][j]==BLACK:
                    black_count+=1
                if game_state.grid[i][j]==WHITE:
                    white_count+=1
            if white_count>black_count:
                white_score+=1
            elif white_count<black_count:
                black_score+=1
        
        return black_score if color==BLACK else white_score
    

#Performs the best when depth=3, agent=Minimaxagent
class WeightedMajorityDifference(BaseHeuristic):
    """
    if a row has weighted score of Black > weighted score of White, black_score+=|weighted score of Black - weighted score of White|
    if a row has weighted score of Black < weighted score of White, white_score+=|weighted score of Black - weighted score of White|
    """
    def evaluate(self, game_state, color) -> float:
        black_score=0
        white_score=0
        for i in range(BOARD_WIDTH):
            black_count=0
            white_count=0
            for j in range(BOARD_WIDTH):
                if game_state.grid[i][j]==BLACK:

                    if (i==0 or i==BOARD_WIDTH-1) and (j==0 or j==BOARD_WIDTH-1):
                        black_count+=3
                    elif (i,j) in star_positions:
                        pass
                    elif (i==0 or i==BOARD_WIDTH-1) or (j==0 or j==BOARD_WIDTH-1):
                        black_count+=2
                    else:
                        black_count+=1
                if game_state.grid[i][j]==WHITE:
                    if (i==0 or i==BOARD_WIDTH-1) and (j==0 or j==BOARD_WIDTH-1):
                        white_count+=3
                    elif (i,j) in star_positions:
                        pass
                    elif (i==0 or i==BOARD_WIDTH-1) or (j==0 or j==BOARD_WIDTH-1):
                        white_count+=2
                    else:
                        white_count+=1
            if white_count>black_count:
                white_score+=white_count-black_count
            elif white_count<black_count:
                black_score+=black_count-white_count
        
        return black_score if color==BLACK else white_score