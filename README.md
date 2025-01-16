# README

## List of external tools used

- NumPy.
- PyTorch for `DeepLearningAgent`.
- PyGame for grapic display.
- `tqdm` for progress bar.
- Decorator class `lazy_property` under `utils.py` is modified from [this link](https://zhuanlan.zhihu.com/p/76533988)

**Disclaimer**: Although all the code, except for the items listed above, was written by ourselves, the design of our framework's API references the Pacman project from the CS181 course project comes from the University of California, Berkeley, and most of the variable names are interoperable.  Additionally, some parts of the code were copied from the homework solutions we wrote for the aforementioned project.

## Usage

Run the program using:
`python othello.py -1 <agent1> [args] -2 <agent2> [args]`

Here, `<agent1>` and `<agent2>` can be `Player`, `RandomAgent`, or any other non-abstract class that inherits from `BaseAgent` (in `agents.py`). If an agent's `__init__` function has arguments other than `color`, it must be passed in as `key=value` (e.g. `depth=2`, `heuristic=CountOwnPieces`) at `[args]`.

Optional arguments:

- Use `--no-graphics` (`-q`) to disable the graphical user interface.
- Use `--seed <seed>` to set a specific random seed for reproducibility.
- Use `--total-games <number>` (`-n`) to play multiple rounds.
- Use `--num-processors <number>` (`-N`) to perform multiprocessing.

## Training

We've already provided models for `DeepLearningAgent` model and `ApproximateQAgent` model under folder `saves`. If you want to train the models on your own, you can

- Run `python train.py` to train the `DeepLearningAgent` model.
- Run `python q_train.py -1 <agent1> [args] -2 <agent2> [args] -n <num_episodes> --save-path <save_path>` to train the `ApproximateQAgent` model. Here, one of the `<agent1>` and `<agent2>` should be the `ApproximateQAgent`, as the other the adversarial opponent.
