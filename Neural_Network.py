from preprocess import preprocess
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import pandas as pd


class Net(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim):
        super(Net, self).__init__()
        self.num_hidden = num_hidden

        self.fc = nn.Linear(input_dim, hidden_dim[0])
        self.fc_list = []

        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(hidden_dim[-1], output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

    def forward(self, x):
        x = torch.relu(self.fc(x))
        for i in range(self.num_hidden - 1):
            x = torch.relu(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x


seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
train_data_name = 'train.csv'
test_data_name = 'test.csv'
train_data, test_data = preprocess(train_data_name, test_data_name)


train_data = np.mat(train_data)

x_train = train_data[:, 0:-1]
y_train = train_data[:, -1]
x_test = np.mat(test_data)


device = torch.device("cpu")
x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
x_test = torch.FloatTensor(x_test).to(device)


num_hidden = 2
hidden_dim = [30, 30]
dim = x_train.shape[1]
output_dim = 1
net = Net(num_hidden, hidden_dim, dim, output_dim)

step_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
optimizer = torch.optim.SGD(net.parameters(), lr=step_lr, momentum=momentum, weight_decay=weight_decay)

loss_func = nn.MSELoss()


ntrain = x_train.shape[0]


index = np.arange(ntrain)
subn = 100
num_epochs = 1000


for epoch in range(num_epochs):
    np.random.shuffle(index)
    for iter_index in range(ntrain // subn):
        subsample = index[(iter_index * subn):((iter_index + 1) * subn)]
        optimizer.zero_grad()
        loss = loss_func(net(x_train[subsample,]), y_train[subsample,])
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        print('epoch: ', epoch)
        output = net(x_train)
        train_loss = loss_func(output, y_train)
        print("train loss: ", train_loss)

with torch.no_grad():
    y_test_predict = torch.exp(net(x_test)) - 1


test_predict_df = pd.DataFrame({'Id':test_data.index, 'SalePrice':y_test_predict.data.cpu().view(-1).numpy()})

submit_file_name = 'neural_network_submission.csv'
test_predict_df.to_csv(path_or_buf=submit_file_name, index = False)