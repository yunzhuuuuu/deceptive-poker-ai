"""
Evaluate trained poker agents against different opponents

1. Loads trained models (with different lambda_entropy values)
2. Plays them against various opponents (random, rule-based, CFR, each other)
3. Measures performance and behavioral metrics
"""

import numpy as np
import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import tournament, set_seed
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Import the model architecture
import sys
sys.path.append('.')
from train import PokerNet, Config as TrainConfig


# Configuration

class Config:
    model_dir = './models'
    env_name = 'limit-holdem' 

    # Specify dataset type and lambda values
    dataset_type = 'unbalanced'  # balanced/unbalanced
    lambda_values = [0, 0.1, 0.3, 0.5, 0.8, 1]
    
    # Evaluation parameters
    num_eval_games = 1000  # Games per matchup
    num_behavior_games = 500  # Games for behavioral metrics
    
    # Output
    results_dir = './results'
    
    # Reproducibility
    seed = 42


# Neural Network Agent Wrapper

class NeuralAgent:
    """Wrapper to use trained PyTorch model as RLCard agent"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.use_raw = False
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        args = checkpoint['args']
        
        # Get model dimensions from checkpoint
        model_state = checkpoint['model_state_dict']
        input_dim = model_state['network.0.weight'].shape[1]
        num_actions = model_state[list(model_state.keys())[-1]].shape[0]
        
        # Create model
        self.model = PokerNet(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dims=getattr(args, 'hidden_dims', TrainConfig.hidden_dims)
        ).to(device)
        
        self.model.load_state_dict(model_state)
        self.model.eval()
        
    
    def eval_step(self, state):
        """
        Predict action for given state
        
        Args:
            state: RLCard state dict with 'obs' and 'legal_actions'
        
        Returns:
            action: Selected action
            info: Dictionary with action probabilities
        """
        # Convert state to tensor
        obs = state['obs']
        features = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.model(features).cpu().numpy()[0]
        
        # Mask illegal actions
        legal_actions = list(state['legal_actions'].keys())
        masked_probs = np.zeros_like(action_probs)
        masked_probs[legal_actions] = action_probs[legal_actions]
        
        # Renormalize
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            # Fallback to uniform over legal actions
            masked_probs[legal_actions] = 1.0 / len(legal_actions)
        
        # Sample action
        action = np.random.choice(len(masked_probs), p=masked_probs)
        
        # Build info dict
        info = {}
        # Map legal action indices to their names and probabilities
        info['probs'] = {
            state['raw_legal_actions'][list(state['legal_actions'].keys()).index(i)]: float(masked_probs[i]) 
            for i in legal_actions
        }

        return action, info


# Behavioral Metrics

def compute_action_entropy(agent, env, num_games=500):
    """
    Measure how unpredictable the agent's actions are
    
    Higher entropy = more mixed strategies = harder to exploit
    """
    entropies = []
    
    for _ in range(num_games):
        state, _ = env.reset()
        
        while not env.is_over():
            current_player = env.get_player_id()
            
            if current_player == 0:
                _, info = agent.eval_step(state)
                probs = np.array(list(info['probs'].values()))
                
                # Compute entropy
                probs_clean = probs[probs > 0]
                if len(probs_clean) > 1:
                    entropy = -np.sum(probs_clean * np.log(probs_clean))
                    entropies.append(entropy)
                    
            # Step environment
            action = agent.eval_step(state)[0] if current_player == 0 else np.random.choice(env.num_actions)
            state, _ = env.step(action)
    
    return np.mean(entropies) if entropies else 0.0


def compute_action_diversity(agent, env, num_games=500):
    """
    Measure diversity of actions taken
    
    Returns fraction of time each action is taken
    """
    action_counts = defaultdict(int)
    total_actions = 0
    
    for _ in range(num_games):
        state, _ = env.reset()
        
        while not env.is_over():
            current_player = env.get_player_id()
            
            if current_player == 0:
                action, info = agent.eval_step(state)

                # Convert action index back to action name
                actual_action_name = state['raw_legal_actions'][list(state['legal_actions'].keys()).index(action)]
                action_counts[actual_action_name] += 1
                total_actions += 1
            else:
                action = np.random.choice(env.num_actions)
            
            state, _ = env.step(action)
            
    if total_actions > 0:
        action_fractions = {
            action: count / total_actions 
            for action, count in action_counts.items()
        }
    else:
        action_fractions = {}
    
    return action_fractions


# Evaluation

def evaluate_agent(agent, env, opponent, num_games=1000):
    """
    Play agent against opponent and return win rate
    
    Returns:
        Average payoff per game (positive = winning, negative = losing)
    """
    env.set_agents([agent, opponent])
    payoffs = []
    
    for _ in range(num_games):
        _, game_payoffs = env.run(is_training=False)
        payoffs.append(game_payoffs[0])
    
    return np.mean(payoffs)


def find_models(model_dir, lambda_values, dataset_type=None):
    """
    Find model files based on dataset type
    
    Args:
        model_dir: Directory containing model files
        lambda_values: List of lambda values to find
        dataset_type: 'balanced', 'unbalanced', or None for auto-detect
    
    Returns:
        Dictionary mapping lambda values to file paths and detected dataset type
    """
    model_paths = {}
    detected_type = None
    
    for lambda_val in lambda_values:
        # Try balanced first
        balanced_path = os.path.join(model_dir, f'best_model_lambda_{lambda_val}_balanced.pt')
        unbalanced_path = os.path.join(model_dir, f'best_model_lambda_{lambda_val}.pt')
        
        if dataset_type == 'balanced':
            if os.path.exists(balanced_path):
                model_paths[lambda_val] = balanced_path
                detected_type = 'balanced'
        elif dataset_type == 'unbalanced':
            if os.path.exists(unbalanced_path):
                model_paths[lambda_val] = unbalanced_path
                detected_type = 'unbalanced'
        else:  # Auto-detect
            if os.path.exists(balanced_path):
                model_paths[lambda_val] = balanced_path
                detected_type = 'balanced'
            elif os.path.exists(unbalanced_path):
                model_paths[lambda_val] = unbalanced_path
                detected_type = 'unbalanced'
    
    return model_paths, detected_type


def compare_models(config):
    """
    Compare multiple models against each other and baselines
    """
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Setup environment
    env = rlcard.make(config.env_name, config={'seed': config.seed})
    set_seed(config.seed)
    
    # Find model paths
    model_paths, detected_type = find_models(
        config.model_dir, 
        config.lambda_values, 
        config.dataset_type
    )
    
    if not model_paths:
        print(f"No models found in {config.model_dir}")
        print(f"Looking for lambda values: {config.lambda_values}")
        print(f"Dataset type: {config.dataset_type or 'auto-detect'}")
        return None
    
    print(f"\nDataset type: {detected_type}")
    print(f"Found {len(model_paths)} models:")
    for lambda_val, path in sorted(model_paths.items()):
        print(f"  λ={lambda_val}: {path}")
    
    # Load all models
    agents = {}
    for lambda_val, path in sorted(model_paths.items()):
        agents[f'λ={lambda_val}'] = NeuralAgent(path)
    
    # Add baseline agents
    agents['Random'] = RandomAgent(num_actions=env.num_actions)
    
    # Try to load CFR if available
    try:
        from rlcard import models
        if config.env_name == 'leduc-holdem':
            cfr_agent = models.load('leduc-holdem-cfr').agents[0]
            agents['CFR'] = cfr_agent
            print("\nLoaded CFR baseline")
    except:
        print("\nCFR baseline not available")
    
    # Results storage
    results = defaultdict(dict)
    results['_meta'] = {'dataset_type': detected_type}
    
    print(f"\nEvaluating {len([k for k in agents.keys() if k.startswith('λ=')])} agents with {config.num_eval_games} games each...")
    
    # Play each agent against baselines
    for agent_name, agent in agents.items():
        if not agent_name.startswith('λ='):
            continue
            
        print(f"\nAgent: {agent_name}")
        
        # Against random
        payoff = evaluate_agent(agent, env, agents['Random'], config.num_eval_games)
        results[agent_name]['vs_Random'] = payoff
        print(f"  vs Random: {payoff:+.4f}")
        
        # Against CFR
        if 'CFR' in agents:
            payoff = evaluate_agent(agent, env, agents['CFR'], config.num_eval_games)
            results[agent_name]['vs_CFR'] = payoff
            print(f"  vs CFR: {payoff:+.4f}")
        
        # Behavioral metrics
        entropy = compute_action_entropy(agent, env, num_games=config.num_behavior_games)
        diversity = compute_action_diversity(agent, env, num_games=config.num_behavior_games)
        
        results[agent_name]['entropy'] = entropy
        results[agent_name]['diversity'] = diversity
        
        print(f"  Entropy: {entropy:.3f}")
        print(f"  Action diversity: {diversity}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY - {detected_type.upper()} DATASET")
    
    # Compare performance vs random
    neural_agents = {k: v for k, v in results.items() if k.startswith('λ=')}
    
    print("\nPerformance vs Random (higher is better):")
    for agent_name in sorted(neural_agents.keys(), 
                             key=lambda x: results[x].get('vs_Random', -999), 
                             reverse=True):
        if 'vs_Random' in results[agent_name]:
            print(f"  {agent_name:15s}: {results[agent_name]['vs_Random']:+.4f}")
    
    # Compare behavioral metrics
    if neural_agents:
        print("\nBehavioral Metrics:")
        print(f"  {'Agent':15s} {'Entropy':>10s} {'Diversity':>50s}")
        print("-" * 100)
        for agent_name in sorted(neural_agents.keys(), key=lambda x: float(x.split('=')[1])):
            if 'entropy' in results[agent_name]:
                diversity_str = str({k: f'{v:.2f}' for k, v in results[agent_name]['diversity'].items()})
                print(f"  {agent_name:15s} {results[agent_name]['entropy']:>10.3f} {diversity_str:>50s}")
    
    return results


def plot_results(results, results_dir):
    """Plot all evaluation results"""
    os.makedirs(results_dir, exist_ok=True)
    
    dataset_type = results.get('_meta', {}).get('dataset_type', 'unknown')
    neural_results = {k: v for k, v in results.items() if k.startswith('λ=')}
    
    if not neural_results:
        print("\nNo neural network results to plot")
        return
    
    # Sort by lambda value
    sorted_keys = sorted(neural_results.keys(), key=lambda x: float(x.split('=')[1]))
    sorted_lambdas = [float(k.split('=')[1]) for k in sorted_keys]
    
    # Plot 1: Performance and Entropy curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Performance vs Random
    if 'vs_Random' in list(neural_results.values())[0]:
        perf_random = [neural_results[k]['vs_Random'] for k in sorted_keys]
        
        axes[0].plot(sorted_lambdas, perf_random, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Lambda (entropy weight)', fontsize=12)
        axes[0].set_ylabel('Payoff vs Random', fontsize=12)
        axes[0].set_title(f'Performance vs Random Agent ({dataset_type})', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Break-even')
        axes[0].legend()
    
    # Entropy
    if 'entropy' in list(neural_results.values())[0]:
        entropies = [neural_results[k]['entropy'] for k in sorted_keys]
        
        axes[1].plot(sorted_lambdas, entropies, marker='o', color='green', linewidth=2, markersize=8)
        axes[1].set_xlabel('Lambda (entropy weight)', fontsize=12)
        axes[1].set_ylabel('Average Action Entropy', fontsize=12)
        axes[1].set_title(f'Behavioral Unpredictability ({dataset_type})', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, f'performance_entropy_{dataset_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformance/Entropy plot saved to: {save_path}")
    plt.close()
    
    # Plot 2: Action Diversity Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    actions = ['call', 'raise', 'fold', 'check']
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#2ecc71']
    
    x = np.arange(len(sorted_keys))
    width = 0.2
    
    for i, action in enumerate(actions):
        values = []
        for agent_name in sorted_keys:
            diversity = neural_results[agent_name].get('diversity', {})
            values.append(diversity.get(action, 0))
        
        ax.bar(x + i * width, values, width, label=action.capitalize(), color=colors[i])
    
    ax.set_xlabel('Agent (Lambda)', fontsize=12)
    ax.set_ylabel('Action Frequency', fontsize=12)
    ax.set_title(f'Action Diversity Across Models ({dataset_type})', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(sorted_keys)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, f'action_diversity_{dataset_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Action diversity plot saved to: {save_path}")
    plt.close()
    
    # Plot 3: Combined Entropy and Diversity (stacked)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bottom values for stacking
    bottom_vals = np.zeros(len(sorted_keys))
    
    for i, action in enumerate(actions):
        values = []
        for agent_name in sorted_keys:
            diversity = neural_results[agent_name].get('diversity', {})
            values.append(diversity.get(action, 0))
        
        ax.bar(sorted_keys, values, bottom=bottom_vals, label=action.capitalize(), color=colors[i])
        bottom_vals += np.array(values)
    
    # Overlay entropy as line
    ax2 = ax.twinx()
    entropies = [neural_results[k]['entropy'] for k in sorted_keys]
    ax2.plot(sorted_keys, entropies, 'ko-', linewidth=2, markersize=8, label='Entropy')
    ax2.set_ylabel('Entropy', fontsize=12)
    
    ax.set_xlabel('Agent (Lambda)', fontsize=12)
    ax.set_ylabel('Action Distribution', fontsize=12)
    ax.set_title(f'Action Diversity and Entropy ({dataset_type})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, f'diversity_entropy_{dataset_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Combined plot saved to: {save_path}")
    plt.close()


# Main

if __name__ == '__main__':
    config = Config()
    results = compare_models(config)    
    plot_results(results, config.results_dir)

    print("Evaluation complete! Results saved to:", config.results_dir)
