import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def expand_time_series(data, n_in, n_out):
    """
    data: nd array
    n_in: input sequence size
    n_out: output sequence size
    """
    n_features = data.shape[1]
    res = data
    right = data
    for i in range(n_in + n_out - 1):
        right = np.roll(right, -1, axis=1)
        right[-1, :] = np.array( [np.nan] *  n_features )
        res = np.concatenate((res, right), axis=1)

    df = pd.DataFrame(res)
    df.dropna(inplace=True)
    return df


def data_transform(n_hours=3, n_pred=1):
    """
    n_hours: input sequence size
    n_pred: output sequence size
    """
    df = pd.read_csv('../tensorflow/pollution.csv', header=0, index_col=0)
    values = df.values
    # process wind direction column
    values[:, 4] = LabelEncoder().fit_transform(values[:, 4])
    # normalize all data into range(0, 1)
    values = MinMaxScaler((0, 1)).fit_transform(values)

    n_features = values.shape[1]
    expand_table = expand_time_series(values, n_hours, n_pred)
    return expand_table.values

class RNN(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1
    ):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.D = 1 if num_layers == 1 else 2 
        # x -> (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size)
        output, hn = self.rnn(x, h0)
        last_output = output[:, -1, :]
        pred = self.fc(last_output)
        return pred

class LSTM(nn.Module):
    def __init__(self, 
        input_size,
        hidden_size,
        output_size,
        num_layers
    ):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x -> (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.hidden_cell = (
            # torch.zeros(1, , self.hidden_layer_size).to('cuda'),
            # torch.zeros(1, , self.hidden_layer_size).to('cuda'),
        # )

    def forward(self, x, bidirectional=False):
        if bidirectional:
            D = 2
        else:
            D = 1
        
        h0 = torch.zeros(D * self.num_layers, x.size(0), self.hidden_size).to('cuda')
        c0 = torch.zeros(D * self.num_layers, x.size(0), self.hidden_size).to('cuda')

        output, (hn, cn) = self.lstm(
            x, 
            (h0, c0)
        )
        last_output = output[:, -1, :]
        pred = self.fc(last_output)
        return pred

class PollutionDataset(Dataset):

    def __init__(self, n_hours, n_pred, 
        n_features=8, train=True, transform=None, test_size=0.3):
        raw_data = data_transform(n_hours, n_pred)
        n_samples = len(raw_data)
        row_boundary = int(n_samples * (1 - test_size))
        if train:
            trgt_data = raw_data[: row_boundary, :]
        else:
            trgt_data = raw_data[row_boundary : , :]
        boundary = n_hours * n_features
        X, y = trgt_data[: , : boundary], trgt_data[:, boundary :]
        self.X = torch.Tensor(X).reshape((-1, n_hours, n_features))
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        


def train(epoch, train_dataloader, model, criterion, optimizer, device):
    model.train()
    for batch_idx, (seq, labels) in enumerate(train_dataloader):
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Epoch: {}  Loss: {}".format(epoch, loss.item()))

    return loss.item()
    
        
    
def test(epoch, test_dataloader, model, criterion, device, verbose=False):
    model.eval()
    val_loss = 0
    with torch.no_grad(): 
        for seq, labels in test_dataloader:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            cur_loss = criterion(y_pred, labels).item()
            val_loss += cur_loss

    avg_loss = val_loss / len(test_dataloader)
    if verbose:
        print("Test Loss at epoch {}: {}".format(epoch, avg_loss))
    return avg_loss


def plot_single_epoch(train_loader, model, optimizer, device, criterion, test_loader):
    loss_hist = [[], []] 
    for batch_idx, (seq, labels) in enumerate(train_loader):
        # print("Batch: " + str(batch_idx))
        model.train()
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred = model(seq)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        test_loss = test(1, test_loader, model, criterion, device)

        loss_hist[0].append(loss.item())
        loss_hist[1].append(test_loss)

    x = range(len(train_loader))
    plt.plot(x, loss_hist[0], color='red', label='train_loss')
    plt.plot(x, loss_hist[1], color='green', label='test_loss')
    plt.legend(loc='upper right')
    plt.savefig('plots/train_loss_air_lstm.jpg')
    print("Figure saved.")
    

def main():
    n_hours = 3
    n_pred = 1
    n_features = 8
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_set = PollutionDataset(n_hours, n_pred, train=True)
    test_set = PollutionDataset(n_hours, n_pred, train=False)
    train_loader = DataLoader(train_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)


    model = LSTM(
        input_size = n_features,
        hidden_size = 256,
        output_size = n_features,
        num_layers = 1
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    epochs = 5

    loss_hist = [[], []] 
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, train_loader, model, criterion, optimizer, device)
        print()
        test_loss = test(epoch, test_loader, model, criterion, device, verbose=True)
        print("\n")
        loss_hist[0].append(train_loss)
        loss_hist[1].append(test_loss)




if __name__ == "__main__":
    main()