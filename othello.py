import argparse
from agents import BaseAgent, Player
from game import Game
from game_state import GameStatus
from utils import *
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def get_agent_from_cli(cli_list: list[str], color: ColorType) -> BaseAgent:
    agent_name = cli_list[0]
    args = cli_list[1:]
    kwargs = {}
    for arg in args:
        k, v = arg.split("=")
        if k == 'depth':
            v = int(v)
        elif k == 'heuristic':
            pass
        else:
            raise ValueError(f"Unknown arg {k}={v}.")
        kwargs[k] = v
    return BaseAgent.registry[agent_name.lower()](color, **kwargs)


def play_game(args):
    agent1, agent2, graphics, debug, seed = args
    seed_everything(seed)
    game = Game(agent1, agent2, graphics, debug)
    return game.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''
Othello for CS181 Project.

Add copyright information here.
'''
    )
    parser.add_argument(
        '--agent1', '-1', nargs="+", type=str, default=['Player']
    )
    parser.add_argument(
        '--agent2', '-2', nargs="+", type=str, default=['RandomAgent']
    )
    parser.add_argument('--no-graphics', '-q', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--total-games', '-n', type=int, default=1)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--num-processors', '-N', type=int, default=1)

    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        seed = random.getrandbits(32)

    agent1 = get_agent_from_cli(args.agent1, 'B')
    agent2 = get_agent_from_cli(args.agent2, 'W')

    total_games: int = args.total_games
    graphics: bool = not args.no_graphics
    debug: bool = args.debug

    if (
        total_games > 1 and
        not isinstance(agent1, Player) and
        not isinstance(agent2, Player)
    ):
        graphics = False

    if total_games == 1:
        final_state = play_game(
            (agent1, agent2, graphics, debug, seed)
        )
        black_count, white_count = final_state.black_white_counts
        print(f'{final_state}')
        print(f'Black {black_count}-{white_count} White')

    else:
        num_processors: int = args.num_processors

        if debug or isinstance(agent1, Player) or isinstance(agent2, Player):
            num_processors = 1

        from multiprocessing import Pool, Manager
        from tqdm import tqdm

        with Manager() as manager:
            shared_results = manager.dict({
                GameStatus.BLACK_WIN: 0,
                GameStatus.WHITE_WIN: 0,
                GameStatus.DRAW: 0
            })
            with Pool(processes=num_processors) as pool:
                args_list = [
                    (agent1, agent2, graphics, debug, i+seed)
                    for i in range(total_games)
                ]
                with tqdm(total=total_games, desc="0-0-0") as pbar:
                    for result in pool.imap_unordered(play_game, args_list):
                        shared_results[result.status] += 1
                        black_wins = shared_results[GameStatus.BLACK_WIN]
                        white_wins = shared_results[GameStatus.WHITE_WIN]
                        draw = shared_results[GameStatus.DRAW]
                        pbar.set_description(
                            f"{black_wins}-{draw}-{white_wins}")
                        pbar.update(1)

            black_wins = shared_results[GameStatus.BLACK_WIN]
            white_wins = shared_results[GameStatus.WHITE_WIN]
            draw = shared_results[GameStatus.DRAW]
            print("Game Statistics:")
            print(f"   Black      Tie    White    Total")
            print(f"{black_wins:8} {draw:8} {white_wins:8} {total_games:8}")
