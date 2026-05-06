"""
Train a supervised poker agent using CFR data with entropy regularization

1. Loads CFR-generated training data
2. Trains a neural network to imitate CFR policy (supervised learning)
3. Optionally adds entropy regularization to encourage unpredictability
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Configuration

class Config:
    # Data
    data_path = "./data/cfr_dataset_5000eps_balanced.pkl"
    # use the balanced dataset alternatively, remember to change output names
    val_split = 0.1

    # Model
    hidden_dims = [32, 16]

    # Training
    epochs = 50
    batch_size = 256
    lr = 0.001
    lambda_entropy_list = [0, 0.1, 0.3, 0.5, 0.8, 1] 

    seed = 42
    save_dir = "./models"
    result_save_dir = "./results/training_curves"
    print_every = 10


# Dataset

class PokerDataset(Dataset):
    """Dataset for poker state-action pairs from CFR"""
    
    def __init__(self, data_path):
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples")

        # Reverse action order from (check, fold, raise, call) to (call, raise, fold, check)
        for sample in self.data:
            sample['action_probs'] = sample['action_probs'][[3, 2, 1, 0]]

        # Relabel based on argmax
        action_names = ['call', 'raise', 'fold', 'check']
        for sample in self.data:
            argmax_idx = sample['action_probs'].argmax()
            sample['action_taken'] = action_names[argmax_idx]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'features': torch.FloatTensor(sample['features']),
            'action_probs': torch.FloatTensor(sample['action_probs']),
        }


# Model

class PokerNet(nn.Module):
    """Neural network for poker policy"""
    
    def __init__(self, input_dim, num_actions, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, return_logits=False):
        """
        Args:
            x: Input features (batch_size, input_dim)
            return_logits: If True, return raw logits instead of probabilities
        
        Returns:
            Action probabilities (batch_size, num_actions) or logits
        """
        logits = self.network(x)
        
        if return_logits:
            return logits
        else:
            return self.softmax(logits)


# Loss Functions

def cross_entropy_loss(pred_probs, target_probs, epsilon=1e-10):
    """
    Cross-entropy loss (equivalent to negative log-likelihood)
    H(target, pred) = -sum(target * log(pred))
    
    Standard loss for imitating a probability distribution
    """
    loss = -torch.sum(target_probs * torch.log(pred_probs + epsilon), dim=-1)
    return loss.mean()


def entropy_regularization(pred_probs, target_probs, epsilon=1e-10):
    """
    Entropy regularization with bluff-aware weighting
    
    Estimate hand strength from target_probs:
    - Strong hands → high aggressive action probability → reduce entropy (stay decisive)
    - Weak hands → low aggressive probability → encourage HIGH entropy (bluffing)
    """
    # (call, raise, fold, check)
    ACTION_WEIGHTS = torch.tensor([0.5, 1, 0, 0.5], device=pred_probs.device)

    # Estimate "hand strength"
    aggressive_score = torch.sum(target_probs * ACTION_WEIGHTS, dim=-1)

    # Weak hand → weight close to 1, strong hand → weight close to 0
    bluff_weight = 1.0 - aggressive_score.detach()  # detach so we don't backprop through targets

    # Standard entropy
    entropy = -torch.sum(pred_probs * torch.log(pred_probs + epsilon), dim=-1)

    weighted_entropy = bluff_weight * entropy

    return weighted_entropy.mean()


def combined_loss(pred_probs, target_probs, lambda_entropy=0.0):
    """
    Combined loss: imitation + entropy regularization
    Loss = imitation_loss - lambda_entropy * entropy
    
    Args:
        pred_probs: Model's predicted action probabilities
        target_probs: CFR's action probabilities (ground truth)
        lambda_entropy: Weight for entropy term (0 = pure imitation, >0 = more random)
    
    Returns:
        Total loss (scalar)
    """
    # Imitation loss (standard cross-entropy)
    imitation_loss = cross_entropy_loss(pred_probs, target_probs)
    
    # Entropy of predictions (we want to maximize this, so subtract it)
    entropy = entropy_regularization(pred_probs, target_probs)
    
    # Combined loss
    total_loss = imitation_loss - lambda_entropy * entropy
    
    return total_loss, imitation_loss, entropy


# Training

def train_epoch(model, dataloader, optimizer, lambda_entropy, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_imitation = 0
    total_entropy = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        target_probs = batch['action_probs'].to(device)
        
        # Forward pass
        pred_probs = model(features)
        
        # Compute loss
        loss, imitation, entropy = combined_loss(pred_probs, target_probs, lambda_entropy)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_imitation += imitation.item()
        total_entropy += entropy.item()
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'imitation': total_imitation / num_batches,
        'entropy': total_entropy / num_batches,
    }


def evaluate(model, dataloader, lambda_entropy, device):
    """Evaluate on validation set"""
    model.eval()
    
    total_loss = 0
    total_imitation = 0
    total_entropy = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            target_probs = batch['action_probs'].to(device)
            
            pred_probs = model(features)
            loss, imitation, entropy = combined_loss(pred_probs, target_probs, lambda_entropy)
            
            total_loss += loss.item()
            total_imitation += imitation.item()
            total_entropy += entropy.item()
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'imitation': total_imitation / num_batches,
        'entropy': total_entropy / num_batches,
    }


def train_model(args, lambda_entropy): 
    """Main training loop"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    dataset = PokerDataset(args.data_path)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Split: {train_size} train, {val_size} val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # Get dimensions from data
    sample = dataset[0]
    input_dim = sample['features'].shape[0]
    num_actions = sample['action_probs'].shape[0]
    
    print(f"Model: Input={input_dim}, Hidden={args.hidden_dims}, Output={num_actions}")
    
    # Create model
    model = PokerNet(
        input_dim=input_dim,
        num_actions=num_actions,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"Training lambda={lambda_entropy} for {args.epochs} epochs...")
    
    history = defaultdict(list)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, lambda_entropy, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, lambda_entropy, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save metrics
        for key in train_metrics:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Train={train_metrics['loss']:.4f}, Val={val_metrics['loss']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = os.path.join(args.save_dir, f'best_model_lambda_{lambda_entropy}_balanced.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': args,
            }, save_path)
    
    # Plot training curves
    plot_training_curves(history, args.result_save_dir, lambda_entropy)
    
    print(f"Completed!")
    print("=" * 70)
    

def plot_training_curves(history, save_dir, lambda_entropy):
    """Plot and save training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['loss', 'imitation', 'entropy']
    titles = ['Total Loss', 'Imitation Loss', 'Entropy']
    
    for ax, metric, title in zip(axes, metrics, titles):
        ax.plot(history[f'train_{metric}'], label='Train', alpha=0.7)
        ax.plot(history[f'val_{metric}'], label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} (λ={lambda_entropy})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_curves_lambda_{lambda_entropy}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Main

if __name__ == '__main__':
    args = Config()

    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_save_dir, exist_ok=True)

    print(f"START TRAINING")
    print(f"Lambda values: {args.lambda_entropy_list}")

    # Train    
    for i, lambda_val in enumerate(args.lambda_entropy_list):
        print(f"\n[MODEL {i+1}/{len(args.lambda_entropy_list)}] λ = {lambda_val}")
        train_model(args, lambda_val)
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print(f"Models saved to: {args.save_dir}")
    print(f"Training curves saved to: {args.result_save_dir}")
