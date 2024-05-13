import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import product


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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
    return total_loss / len(val_loader)

def grid_search(model, param_grid, train_data, val_data):
    grid_list = list(product(*param_grid.values()))
    print(grid_list)
    best_score = float('inf')
    best_params = None
    
    for params_tuple in grid_list:

        params = dict(zip(param_grid.keys(), params_tuple))
        score = train_and_evaluate_model(model, params, train_data, val_data)
        print(f"Tested Params: {params} Score: {score}")

        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score


