import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add these changes to your PhasicPolicyNetwork class:
class PhasicPolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple feed-forward architecture with careful initialization
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Smaller policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Smaller value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize with small weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add input validation
        if torch.isnan(state).any() or torch.isnan(action).any():
            raise ValueError("NaN detected in inputs")
            
        state = state.float()
        action = action.float()
        
        # Normalize inputs more aggressively
        state = state.clamp(-5, 5) 
        action = action.clamp(-1, 1)
        
        x = torch.cat([state, action], dim=-1)
        shared_features = self.shared(x)
        
        policy_out = self.policy_head(shared_features)
        value_out = self.value_head(shared_features)
        
        return policy_out, value_out

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Generate an action given a state."""
        # Create a dummy action tensor filled with zeros
        batch_size = state.shape[0]
        dummy_action = torch.zeros((batch_size, self.action_dim), device=state.device)
        
        # Get policy output
        policy_out, _ = self.forward(state, dummy_action)
        
        if not deterministic:
            # Add small noise during training for exploration
            noise = torch.randn_like(policy_out) * 0.1
            policy_out = policy_out + noise
            
        # Ensure actions are in [-1, 1]
        return torch.clamp(policy_out, -1, 1)

    def get_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get value prediction for a state-action pair."""
        _, value = self.forward(state, action)
        return value

# Modify your training function:
def train_phasic_policy(
    data_dir: str,
    model_save_path: str,
    num_files: int = 1000,
    epochs: int = 100,
    batch_size: int = 128,  # Reduced batch size
    lr: float = 1e-5,  # Further reduced learning rate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Training on device: {device}")
    
    # Initialize model and optimizer
    model = PhasicPolicyNetwork().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.001,  # Increased weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create dataset and dataloader
    dataset = DrivingDataset(data_dir, num_files=num_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
        drop_last=True  # Drop incomplete batches
    )
    
    # Training metrics
    policy_losses = []
    value_losses = []
    best_total_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_policy_loss = 0
        epoch_value_loss = 0
        valid_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (states, actions, rewards, next_states) in enumerate(progress_bar):
            try:
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                
                # Skip batch if any inputs are invalid
                if torch.isnan(states).any() or torch.isnan(actions).any() or \
                   torch.isnan(rewards).any() or torch.isnan(next_states).any():
                    continue
                
                # Get policy and value predictions
                policy_out, value_pred = model(states, actions)
                
                # Skip batch if outputs are invalid
                if torch.isnan(policy_out).any() or torch.isnan(value_pred).any():
                    continue
                
                # Get next state value prediction
                with torch.no_grad():
                    next_actions = model.get_action(next_states)
                    _, next_value = model(next_states, next_actions)
                
                # Calculate advantages with careful clipping
                advantages = rewards + 0.99 * next_value - value_pred
                advantages = torch.clamp(advantages, -1.0, 1.0)  # Tighter clipping
                
                # Calculate losses
                policy_loss = -torch.mean(advantages.detach() * policy_out)
                value_loss = F.smooth_l1_loss(value_pred, rewards + 0.99 * next_value.detach()) * 0.5
                
                # Scale losses
                policy_loss = policy_loss * 0.1  # Scale down policy loss
                value_loss = value_loss * 0.1   # Scale down value loss
                
                # Combined loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Skip if loss is invalid
                if torch.isnan(total_loss):
                    continue
                
                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Tighter gradient clipping
                
                optimizer.step()
                
                # Update metrics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                valid_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'policy_loss': f'{policy_loss.item():.6f}',
                    'value_loss': f'{value_loss.item():.6f}'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if valid_batches == 0:
            print("No valid batches in epoch!")
            continue
            
        # Calculate average losses
        avg_policy_loss = epoch_policy_loss / valid_batches
        avg_value_loss = epoch_value_loss / valid_batches
        total_loss = avg_policy_loss + avg_value_loss
        
        # Record metrics
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        
        print(f"Epoch {epoch+1}: Policy Loss = {avg_policy_loss:.6f}, Value Loss = {avg_value_loss:.6f}")

class DrivingDataset(Dataset):
    def __init__(self, data_dir: str, num_files: int = 1000):
        self.data = []
        data_path = Path(data_dir)
        csv_files = sorted(list(data_path.glob("*.csv")))[:num_files]
        print(f"Looking for CSV files in: {data_path.absolute()}")
        print(f"Found {len(csv_files)} CSV files")

        # Also verify the file contents
        if csv_files:
            print(f"First file contents preview:")
            df = pd.read_csv(csv_files[0])
            print(df.columns.tolist())
        
        # First pass: calculate normalization statistics
        states_sum = np.zeros(4)
        states_sq_sum = np.zeros(4)
        n_samples = 0
        
        print("Loading dataset...")
        for file in tqdm(csv_files, desc="Processing files"):
            try:
                df = pd.read_csv(file)
                if df.empty:
                    continue
                                    
                processed_df = pd.DataFrame({
                    'target_lataccel': df['targetLateralAcceleration'],
                    'roll_lataccel': np.sin(df['roll']) * 9.81,
                    'v_ego': df['vEgo'],
                    'a_ego': df['aEgo'],
                    'steer_command': -df['steerCommand']
                }).dropna()
                
                if len(processed_df) > 0:
                    dt = 0.01  # 100Hz sample rate
                    for i in range(len(processed_df) - 1):
                        state = np.array([
                            processed_df['target_lataccel'].iloc[i],
                            processed_df['roll_lataccel'].iloc[i],
                            processed_df['v_ego'].iloc[i],
                            processed_df['a_ego'].iloc[i]
                        ], dtype=np.float32)
                        
                        action = processed_df['steer_command'].iloc[i]
                        
                        # Calculate costs
                        lataccel_error = (processed_df['roll_lataccel'].iloc[i] - 
                                        processed_df['target_lataccel'].iloc[i]) ** 2
                        lataccel_cost = lataccel_error * 100
                        
                        jerk = ((processed_df['roll_lataccel'].iloc[i+1] - 
                                processed_df['roll_lataccel'].iloc[i]) / dt) ** 2
                        jerk_cost = jerk * 100
                        
                        total_cost = (lataccel_cost * 50) + jerk_cost
                        reward = -total_cost
                        # Add reward normalization
                        reward = np.clip(reward / 1000.0, -1.0, 1.0)  # Scale down the large cost values
                        
                        next_state = np.array([
                            processed_df['target_lataccel'].iloc[i+1],
                            processed_df['roll_lataccel'].iloc[i+1],
                            processed_df['v_ego'].iloc[i+1],
                            processed_df['a_ego'].iloc[i+1]
                        ], dtype=np.float32)
                        
                        if np.all(np.isfinite(state)) and np.all(np.isfinite(next_state)):
                            self.data.append((state, action, reward, next_state))
                            states_sum += state
                            states_sq_sum += state ** 2
                            n_samples += 1
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        # Calculate normalization statistics
        self.state_means = states_sum / n_samples
        self.state_stds = np.sqrt(states_sq_sum / n_samples - self.state_means ** 2)
        self.state_stds = np.where(self.state_stds == 0, 1.0, self.state_stds)
        
        print(f"Successfully loaded {len(self.data)} samples from {len(csv_files)} files")
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        state, action, reward, next_state = self.data[idx]
        
        # Convert to tensors with validation
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        next_state_tensor = torch.FloatTensor(next_state)
        
        # Apply robust preprocessing
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0)
        action_tensor = torch.nan_to_num(action_tensor, nan=0.0)
        reward_tensor = torch.nan_to_num(reward_tensor, nan=0.0)
        next_state_tensor = torch.nan_to_num(next_state_tensor, nan=0.0)
        
        # Clip values to reasonable ranges
        state_tensor = torch.clamp(state_tensor, -10.0, 10.0)
        action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
        next_state_tensor = torch.clamp(next_state_tensor, -10.0, 10.0)
        
        return (state_tensor, action_tensor, reward_tensor, next_state_tensor)
def train_phasic_policy(
    data_dir: str,
    model_save_path: str,
    num_files: int = 1000,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,  # Reduced learning rate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Training on device: {device}")
    
    # Initialize model and optimizer
    model = PhasicPolicyNetwork().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # Added weight decay
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create dataset and dataloader
    dataset = DrivingDataset(data_dir, num_files=num_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Training metrics
    policy_losses = []
    value_losses = []
    best_total_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_policy_loss = 0
        epoch_value_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (states, actions, rewards, next_states) in enumerate(progress_bar):
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            
            # Get policy and value predictions
            policy_out, value_pred = model(states, actions)
            
            # Get next state value prediction
            with torch.no_grad():
                next_actions = model.get_action(next_states)
                _, next_value = model(next_states, next_actions)
            # Calculate advantages with clipping
            advantages = rewards + 0.99 * next_value - value_pred
            advantages = torch.clamp(advantages, -1.0, 1.0)
            
            # Calculate losses with gradient clipping
            policy_loss = -torch.mean(advantages.detach() * policy_out)
            value_loss = F.smooth_l1_loss(value_pred, rewards + 0.99 * next_value.detach())
            
            # Combined loss
            total_loss = policy_loss + 0.2 * value_loss  # Reduce value loss contribution
            # Check for NaN
            if torch.isnan(total_loss):
                print("NaN loss detected, skipping batch")
                continue
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Update metrics
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'policy_loss': f'{policy_loss.item():.4f}',
                'value_loss': f'{value_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        total_loss = avg_policy_loss + avg_value_loss
        
        # Update learning rate
        scheduler.step(total_loss)
        
        # Record metrics
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        
        print(f"Epoch {epoch+1}: Policy Loss = {avg_policy_loss:.4f}, Value Loss = {avg_value_loss:.4f}")
        
        # Save best model
        if total_loss < best_total_loss:
            best_total_loss = total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'total_loss': total_loss,
                'state_means': dataset.state_means,
                'state_stds': dataset.state_stds
            }, f"{model_save_path}_best.pt")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'total_loss': total_loss,
                'state_means': dataset.state_means,
                'state_stds': dataset.state_stds
            }, f"{model_save_path}_epoch_{epoch+1}.pt")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(policy_losses)
    plt.title('Policy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(value_losses)
    plt.title('Value Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f"{model_save_path}_training_curves.png")
    plt.close()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_means': dataset.state_means,
        'state_stds': dataset.state_stds
    }, f"{model_save_path}_final.pt")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a phasic policy model for vehicle control')
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CSV data files")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save model checkpoints")
    parser.add_argument("--num_files", type=int, default=1000, help="Number of CSV files to use for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to use for training (cuda/cpu)")
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    print(f"Starting training with the following parameters:")
    print(f"Data directory: {args.data_dir}")
    print(f"Model save path: {args.model_save_path}")
    print(f"Number of files: {args.num_files}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    
    train_phasic_policy(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        num_files=args.num_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )