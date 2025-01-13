import argparse
from agents import BaseAgent, Player
from game import Game
from game_state import GameStatus
from utils import *


def seed_everything(seed: int) -> None:
    import random

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
        elif k == 'branch':
            v = int(v)
        else:
            raise ValueError(f"Unknown arg {k}={v}.")
        kwargs[k] = v
    return BaseAgent.registry[agent_name.lower()](color, **kwargs)


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

    args = parser.parse_args()

    seed = args.seed
    if seed is not None:
        seed_everything(seed)

    agent1 = get_agent_from_cli(args.agent1, 'B')
    agent2 = get_agent_from_cli(args.agent2, 'W')

    total_games: int = args.total_games
    graphics: bool = not args.no_graphics

    if (
        total_games > 1 and
        not isinstance(agent1, Player) and
        not isinstance(agent2, Player)
    ):
        graphics = False

    if total_games == 1:
        game = Game(agent1, agent2, graphics, args.debug)
        final_state = game.start()
        black_count, white_count = final_state.black_white_counts
        print(f'{final_state}')
        print(f'Black {black_count}-{white_count} White')

    else:
        from tqdm import tqdm

        black_wins = 0
        white_wins = 0
        draw = 0

        pbar = tqdm(range(total_games), desc="0-0-0")
        for _ in pbar:
            game = Game(agent1, agent2, graphics, args.debug)
            final_status = game.start().status
            if final_status == GameStatus.BLACK_WIN:
                black_wins += 1
            elif final_status == GameStatus.WHITE_WIN:
                white_wins += 1
            elif final_status == GameStatus.DRAW:
                draw += 1

            pbar.set_description(f"{black_wins}-{draw}-{white_wins}")

        print("Game Statistics:")
        print(f"   Black      Tie    White    Total")
        print(f"{black_wins:8} {draw:8} {white_wins:8} {total_games:8}")
