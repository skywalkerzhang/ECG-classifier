import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(120, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 8),
        #     nn.ReLU()
        # )
        self.decoder = nn.Sequential(
            # nn.Linear(8, 32),
            # nn.ReLU(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            # nn.Linear(64, 120),
            # nn.ReLU(),
            nn.Linear(120, 3),
            nn.ReLU(),
            nn.Softmax(1)
        )

    def forward(self, X):
        batch_size = X.size(0)
        X = X.view(batch_size, -1)
        # X = self.encoder(X)
        X = self.decoder(X)
        return X

class ECGDataSet(Dataset):
    def __init__(self, feature_path, label_path):
        self.features = torch.from_numpy(np.load(feature_path)).to(torch.float32)
        self.labels = torch.from_numpy(np.load(label_path)).to(torch.float32).T

    def __getitem__(self, index):
        features = self.features[index]
        labels = self.labels[index]
        return features, labels

    def __len__(self):
        return len(self.features)


batch_size = 8
epochs = 6
data = ECGDataSet('./processed_training_sets0.npy', './training_labels0.npy')

train_data = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)

net = Autoencoder()
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
for epoch in range(epochs):
    print(f'epoch {epoch} starting')
    for X, y in train_data:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()

true_count = 0
def predict(net, data_iter):
    global true_count
    for X, y in data_iter:
        preds = net(X).argmax(axis=1)
        for i in range(len(preds)):
            if (y[i][preds[i]] == 1):
                true_count += 1

predict(net, train_data)
print(true_count, '/', len(data))

