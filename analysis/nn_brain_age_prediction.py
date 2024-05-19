import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from torch.optim import Adam


# Create a class representing a brain age prediction model based on 4 layers of NNs
class BrainAgeModel(nn.Module):
    def __init__(self, input_size, size_hidden1, size_hidden2, size_hidden3, size_hidden4):
        super().__init__()
        # First layer: linear transformation from input size to first hidden layer size
        self.lin1 = nn.Linear(input_size, size_hidden1)
        # Activation function for the first hidden layer
        self.relu1 = nn.ReLU()
        # Second layer: linear transformation from first to second hidden layer size
        self.lin2 = nn.Linear(size_hidden1, size_hidden2)
        # Activation function for the second hidden layer
        self.relu2 = nn.ReLU()
        # Third layer: linear transformation from second to third hidden layer size
        self.lin3 = nn.Linear(size_hidden2, size_hidden3)
        # Activation function for the third hidden layer
        self.relu3 = nn.ReLU()
        # Output layer: linear transformation from third hidden layer size to output size
        self.lin4 = nn.Linear(size_hidden3, size_hidden4)

    def forward(self, input):
        # Define the forward pass through the network
        x = self.lin1(input)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        x = self.relu3(x)
        x = self.lin4(x)
        return x


# train model to minimize loss over multiple epochs
def train(model_inp, num_epochs, criterion, learning_rate, train_iter):
    optimizer = torch.optim.RMSprop(model_inp.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # when iterating over train_iter, we split dataset into batches of specified size
        # each iteration yields a batch of X_train and y_train data points
        for inputs, labels in train_iter: # train_iter is a DataLoader obj
            # forward pass
            outputs = model_inp(inputs)
            # defining loss
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            # update weights based on computed gradients
            optimizer.step()
            # accumulating running loss
            running_loss += loss.item()
        # Print loss after each epoch
        print('Epoch [%d]/[%d] running accumulative loss across all batches: %.3f' %
              (epoch + 1, num_epochs, running_loss))
        # Resetting running loss after each epoch
        running_loss = 0.0



def train_and_evaluate_model(model, params, train_data, val_data):
    # Define loss and optimizer
    criterion = nn.MSELoss()
    
    optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

    val_loader = DataLoader(val_data, batch_size=params['batch_size'])

    # Training loop
    for epoch in range(params['num_epochs']):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad() # reset gradients to zereo
            outputs = model(inputs) 
            loss = criterion(outputs, targets) # calculate loss
            loss.backward() # compute gradients of the loss w/r parameters
            optimizer.step() # use computed gradients to adjust params

        # Validation loop
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
    return total_loss / len(val_loader)

def grid_search(input_size, param_grid, train_data, val_data):
    grid_list = list(product(*param_grid.values()))
    best_score = float('inf')
    best_params = None

    
    for params_tuple in grid_list:

        params = dict(zip(param_grid.keys(), params_tuple))
        # Create the model with the parameters from the grid
        model = BrainAgeModel(input_size, 
                              size_hidden1=params['size_hidden1'],
                              size_hidden2=params['size_hidden2'],
                              size_hidden3=params['size_hidden3'],
                              size_hidden4=params['size_hidden4'])

        score = train_and_evaluate_model(model, params, train_data, val_data)
        print(f"Tested Params: {params} Score: {score}")

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score








