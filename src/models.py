import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3, dropout_prob=0.2):
        """
        DQN model.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Number of possible actions.
            hidden_dim (int): Number of neurons in each hidden layer.
            num_layers (int): Number of hidden layers in the network.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(DQN, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_dim, dtype=torch.float64))
        layers.append(nn.BatchNorm1d(hidden_dim, dtype=torch.float64))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64))
            layers.append(nn.BatchNorm1d(hidden_dim, dtype=torch.float64))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_prob))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim, dtype=torch.float64))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Ensure batch norm is properly applied even for a single input
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        return self.network(x)