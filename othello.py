import argparse
from agents import BaseAgent
from game import Game


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def get_agent_from_cli_name(cli_string: str) -> BaseAgent:
    return BaseAgent.registry[cli_string.lower()]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''
Othello for CS181 Project.

Add copyright information here.
'''
    )
    parser.add_argument(
        'agent1', nargs="?", type=str, default='Player'
    )
    parser.add_argument(
        'agent2', nargs="?", type=str, default='RandomAgent'
    )
    parser.add_argument('--no-graphics', action='store_true')
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()

    seed = args.seed
    if seed is not None:
        seed_everything(seed)

    agent1 = get_agent_from_cli_name(args.agent1)
    agent2 = get_agent_from_cli_name(args.agent2)
    game = Game(agent1, agent2, not args.no_graphics)

    game.start()
