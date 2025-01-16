from tqdm import tqdm
import argparse

from agents import ApproximateQAgent
from othello import seed_everything, get_agent_from_cli
from game import Game
from game_state import StateStatus
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Trainer for `ApproximateQAgent`.'''
    )
    parser.add_argument(
        '--agent1', '-1', nargs="+", type=str, default=['ApproximateQAgent']
    )
    parser.add_argument(
        '--agent2', '-2', nargs="+", type=str, default=['RandomAgent']
    )
    parser.add_argument('--save-path', type=str, default='./saves/q-agent.pkl')
    parser.add_argument('--total-games', '-n', type=int, default=1)
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()

    seed = args.seed
    if seed is not None:
        seed_everything(seed)

    agent1 = get_agent_from_cli(args.agent1, BLACK)
    agent2 = get_agent_from_cli(args.agent2, WHITE)

    if isinstance(agent1, ApproximateQAgent):
        agent1.train()
    if isinstance(agent2, ApproximateQAgent):
        agent2.train()

    total_games: int = args.total_games

    black_wins = 0
    white_wins = 0
    draw = 0

    pbar = tqdm(range(total_games), desc="0-0-0")

    for _ in pbar:
        game = Game(agent1, agent2, False)
        final_status = game.start(train=True).status
        if final_status == StateStatus.BLACK_WINS:
            black_wins += 1
        elif final_status == StateStatus.WHITE_WINS:
            white_wins += 1
        elif final_status == StateStatus.GAME_DRAW:
            draw += 1

        pbar.set_description(f"{black_wins}-{draw}-{white_wins}")

    assert not (
        isinstance(agent1, ApproximateQAgent) and
        isinstance(agent2, ApproximateQAgent)
    )

    if isinstance(agent1, ApproximateQAgent):
        agent1.save_model(args.save_path)
    if isinstance(agent2, ApproximateQAgent):
        agent2.save_model(args.save_path)
