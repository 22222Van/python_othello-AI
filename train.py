import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from models import cnn_arch_1, cnn_arch_2, cnn_arch_3, cnn_arch_4, mlp, transformer

class OthelloDataset(Dataset):
    def __init__(self, board_states, actions, current_players):
        self.board_states = torch.tensor(board_states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self.current_players = torch.tensor(current_players, dtype=torch.float32)

    def __len__(self):
        return len(self.board_states)

    def __getitem__(self, idx):
        return self.board_states[idx], self.actions[idx], current_players[idx]

if __name__ == '__main__':

    with open('dataset_medium.json', 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)

    board_states = [dict["state"] for dict in raw_dataset]
    actions = [dict["action"] for dict in raw_dataset]
    current_players = []
    for dict in raw_dataset:
        player = None
        if dict["player"] == "black":
            player = 1
        else:
            player = 0
        current_players.append(player)

    dataset = OthelloDataset(board_states, actions, current_players)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = cnn_arch_2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    st_lr = 1e-3
    ed_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=st_lr)

    num_epochs = 10
    lr_step = (st_lr - ed_lr) / num_epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        new_lr = st_lr - epoch * lr_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        for i, (inputs, labels, players) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            players = players.to(device)
            
            outputs = model(inputs, players)

            outputs = outputs.view(-1, 8*8)
            labels = labels[:, 0] * 8 + labels[:, 1]

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            # if i % 10 == 0:
            #     print(loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

    torch.save(model.state_dict(), "cnn_arch_2.pth")
