import torch
import numpy as np

def XO_to_num(board):
    newboard = torch.zeros(8,8)
    for i in range(8):
        for j in range(8):
            if board[i][j] == 'X':
                newboard[i][j] = 1
            elif board[i][j] == 'O':
                newboard[i][j] = -1
            else:
                newboard[i][j] = 0
    state = newboard
    return state
def get_player(player):
    if player == 'X':
        return 1
    else:
        return 0

def output_action(a):
    letters = "ABCDEFGH"
    action = letters[a[1]] + str(a[0]+1)
    return action

def get_legal_actions(legal_list):
    letter_to_index = {letter: i for i, letter in enumerate("ABCDEFGH")}
    new_list = []
    for action in legal_list:
        x = int(action[1]) - 1
        y = letter_to_index[action[0]]
        new_list.append((x,y))
    return new_list

def predict(model, board_state, current_player, legal_list):
    # board_state(8x8): black 1, white -1, empty 0
    # current_player(1): black 1, white 0
    # legal_list: [(x0,y0), (x1, y1), ...]
    board_state = torch.Tensor(board_state).unsqueeze(0).to(dtype = torch.long)
    current_player = torch.Tensor([current_player]).to(dtype = torch.long)
    
    with torch.no_grad():
        output = model(board_state, current_player)

    output = output.view(-1, 8*8)
    
    legal_values = []
    for move in legal_list:
        row, col = move
        index = row * 8 + col
        legal_values.append((output[0, index].item(), move))

    best_move = max(legal_values, key=lambda x: x[0])[1]
    return best_move