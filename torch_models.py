import torch
import torch.nn as nn


def predict(model, board_state, current_player, legal_list):
    board_state = torch.Tensor(board_state).unsqueeze(0).to(dtype=torch.long)
    current_player = torch.Tensor([current_player]).to(dtype=torch.long)

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


class cnn_arch_1(nn.Module):
    def __init__(self):
        super(cnn_arch_1, self).__init__()
        self.hidden_size = 32
        # black 1, white 0
        self.palyer = nn.Embedding(2, self.hidden_size)  # FIXME: typo
        # black 2, white 0, empty 1
        self.grid = nn.Embedding(3, self.hidden_size)
        self.conv1 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.mlp1 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, p):
        x = self.grid((x + 1).to(dtype=torch.long))
        emb_p = self.palyer(p).unsqueeze(1).unsqueeze(1)
        x = x + emb_p
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x = x.squeeze()
        return x


class cnn_arch_2(nn.Module):
    def __init__(self):
        super(cnn_arch_2, self).__init__()
        self.hidden_size = 32
        # black 1, white 0
        self.palyer = nn.Embedding(2, self.hidden_size)
        # black 2, white 0, empty 1
        self.grid = nn.Embedding(3, self.hidden_size)
        self.conv1 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.mlp1 = nn.Linear(8 * 8 * self.hidden_size, 8 * 8)
        self.relu = nn.ReLU()

    def forward(self, x, p):
        x = self.grid((x + 1).to(dtype=torch.long))
        emb_p = self.palyer(p).unsqueeze(1).unsqueeze(1)
        x = x + emb_p
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1)
        x = self.mlp1(x)
        x = x.reshape(x.size(0), 8, 8)
        return x


class cnn_arch_3(nn.Module):
    def __init__(self):
        super(cnn_arch_3, self).__init__()
        self.hidden_size = 32
        # black 1, white 0
        self.palyer = nn.Embedding(2, self.hidden_size)
        # black 2, white 0, empty 1
        self.grid = nn.Embedding(3, self.hidden_size)
        self.conv1 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.mlp1 = nn.Linear(8 * 8 * self.hidden_size,
                              8 * 8 * self.hidden_size)
        self.mlp2 = nn.Linear(8 * 8 * self.hidden_size, 8 * 8)
        self.relu = nn.ReLU()

    def forward(self, x, p):
        x = self.grid((x + 1).to(dtype=torch.long))
        emb_p = self.palyer(p).unsqueeze(1).unsqueeze(1)
        x = x + emb_p
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.mlp1(x))
        x = self.mlp2(x)
        x = x.reshape(x.size(0), 8, 8)
        return x


class cnn_arch_4(nn.Module):
    def __init__(self):
        super(cnn_arch_4, self).__init__()
        self.hidden_size = 32
        # black 1, white 0
        self.palyer = nn.Embedding(2, self.hidden_size)
        # black 2, white 0, empty 1
        self.grid = nn.Embedding(3, self.hidden_size)
        self.conv1 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.mlp1 = nn.Linear(8 * 8 * self.hidden_size, 8 * 8)
        self.relu = nn.ReLU()

    def forward(self, x, p):
        x = self.grid((x + 1).to(dtype=torch.long))
        emb_p = self.palyer(p).unsqueeze(1).unsqueeze(1)
        x = x + emb_p
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), -1)
        x = self.mlp1(x)
        x = x.reshape(x.size(0), 8, 8)
        return x


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.hidden_size = 8
        self.mlp_hidden_size = self.hidden_size * 8 * 8
        # black 1, white 0
        self.palyer = nn.Embedding(2, self.hidden_size)
        # black 2, white 0, empty 1
        self.grid = nn.Embedding(3, self.hidden_size)
        self.mlp1 = nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size)
        self.mlp2 = nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size)
        self.mlp3 = nn.Linear(self.mlp_hidden_size, 8 * 8)
        self.relu = nn.ReLU()

    def forward(self, x, p):
        x = self.grid((x + 1).to(dtype=torch.long))
        emb_p = self.palyer(p).unsqueeze(1).unsqueeze(1)
        x = x + emb_p
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = self.relu(self.mlp3(x))
        x = x.reshape(x.size(0), 8, 8)
        return x


class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        self.hidden_size = 32
        # black 1, white 0
        self.palyer = nn.Embedding(2, self.hidden_size)
        # black 2, white 0, empty 1
        self.place = nn.Embedding(3, self.hidden_size)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=4, dim_feedforward=self.hidden_size * 4, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=4
        )

        self.mlp1 = nn.Linear(8 * 8 * self.hidden_size, 8 * 8)
        self.relu = nn.ReLU()

    def forward(self, x, p):
        x = self.place((x + 1).to(dtype=torch.long))
        emb_p = self.palyer(p).unsqueeze(1).unsqueeze(1)
        x = x + emb_p
        x = x.view(-1, 8 * 8, self.hidden_size)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.mlp1(x)
        x = x.view(-1, 8, 8)
        return x
