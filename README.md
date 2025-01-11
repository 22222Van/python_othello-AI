# README

## 注意事项

改代码之前先拉取（pull）代码！！！

改代码之前先拉取（pull）代码！！！

改代码之前先拉取（pull）代码！！！

## Usage

Run the program using:
`python othello.py -1 <agent1> [args] -2 <agent2> [args]`

Here, `<agent1>` and `<agent2>` can be `Player`, `RandomAgent`, or any other non-abstract class that inherits from `BaseAgent` (in `agents.py`). You can define additional agents as needed. If an agent's `__init__` function has arguments other than `color`, it must be passed in as `key=value` (e.g. `depth=2`, `heuristic=CountOwnPieces`) at `[args]`.

Optional arguments:

- Use `--no-graphics` to disable the graphical user interface.
- Use `--seed <seed>` to set a specific random seed for reproducibility.
