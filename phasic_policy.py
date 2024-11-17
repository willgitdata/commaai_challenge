import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

VOCAB_SIZE = 1

class PhasicPolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Updated policy head to match the trained model
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)  # Direct output to action space
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        state = torch.where(torch.isnan(state), torch.zeros_like(state), state)
        action = torch.where(torch.isnan(action), torch.zeros_like(action), action)

        x = torch.cat([state, action], dim=-1)
        shared_features = self.shared(x)
        policy_output = self.policy_head(shared_features)
        
        return policy_output, self.value_head(shared_features)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action = torch.zeros((state.shape[0], 1), device=state.device)
        policy_out, _ = self.forward(state, action)
        
        if not deterministic:
            noise = torch.randn_like(policy_out) * 0.1
            policy_out = policy_out + noise
            
        return torch.clamp(policy_out, -1, 1)