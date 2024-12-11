import argparse
from agents import BaseAgent
from game import Game
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


def get_agent_from_cli_name(cli_list: list[str], color: ColorType) -> BaseAgent:
    agent_name = cli_list[0]
    args = cli_list[1:]
    kwargs = {}
    for arg in args:
        k, v = arg.split("=")
        if k == 'depth':
            v = int(v)
        elif k == 'heuristic':
            raise NotImplementedError("Heuristic is not implemented.")
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
    parser.add_argument('--no-graphics', action='store_true')
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()

    seed = args.seed
    if seed is not None:
        seed_everything(seed)

    agent1 = get_agent_from_cli_name(args.agent1, 'B')
    agent2 = get_agent_from_cli_name(args.agent2, 'W')
    game = Game(agent1, agent2, not args.no_graphics)

    game.start()
