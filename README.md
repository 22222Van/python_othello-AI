# README

## Usage

Run the program using:
`python othello.py <agent1> <agent2>`

Here, `<agent1>` and `<agent2>` can be `Player`, `RandomAgent`, or any other non-abstract class that inherits from `BaseAgent`. You can define additional agents as needed.

Optional arguments:

- Use `--no-graphics` to disable the graphical user interface.
- Use `--seed <seed>` to set a specific random seed for reproducibility.

## TODOs

- [ ] The algorithm, especially Minimax, often revisits a same state space at different time steps, so making efficient management of the `GameState` class is crucial to avoid redundant computations. Plans are underway to reconstruct the structure of the `GameState` class, which will result in changes to the API.
