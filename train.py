"""
train.py
--------
Main training script for DRL-based HetNet power optimisation.

Usage:
    python experiments/train.py --config configs/default.yaml
    python experiments/train.py --n_ue 40 --n_sbs 9 --episodes 1000
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.hetnet_env import HetNetEnv
from src.agents.ddpg_agent import DDPGAgent


# ======================================================================
# Argument parser
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train DRL agent for 5G HetNet power optimisation'
    )
    parser.add_argument('--config',   type=str,  default='configs/default.yaml')
    parser.add_argument('--n_ue',     type=int,  default=None)
    parser.add_argument('--n_sbs',    type=int,  default=None)
    parser.add_argument('--episodes', type=int,  default=None)
    parser.add_argument('--algorithm',type=str,  default='ddpg',
                        choices=['ddpg', 'ppo'])
    parser.add_argument('--seed',     type=int,  default=42)
    parser.add_argument('--output',   type=str,  default='results/')
    return parser.parse_args()


# ======================================================================
# Load config
# ======================================================================

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ======================================================================
# Plot utilities
# ======================================================================

def plot_training(rewards, powers, ee_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(rewards, alpha=0.4, color='steelblue', label='raw')
    axes[0].plot(moving_avg(rewards, 50), color='navy', label='MA-50')
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].legend()

    axes[1].plot(powers, alpha=0.4, color='coral', label='raw')
    axes[1].plot(moving_avg(powers, 50), color='darkred', label='MA-50')
    axes[1].set_title('Network Power Consumption (W)')
    axes[1].set_xlabel('Episode')
    axes[1].legend()

    axes[2].plot(ee_list, alpha=0.4, color='mediumseagreen', label='raw')
    axes[2].plot(moving_avg(ee_list, 50), color='darkgreen', label='MA-50')
    axes[2].set_title('Energy Efficiency (bits/Joule)')
    axes[2].set_xlabel('Episode')
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")


def moving_avg(data, window):
    return [np.mean(data[max(0, i-window):i+1])
            for i in range(len(data))]


# ======================================================================
# Training loop
# ======================================================================

def train(cfg, reward_cfg, train_cfg, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environment
    env = HetNetEnv(cfg, reward_cfg)
    print(f"\n{'='*60}")
    print(f"  5G HetNet DRL Power Optimisation")
    print(f"{'='*60}")
    print(f"  UEs     : {cfg['n_ue']}")
    print(f"  s-BSs   : {cfg['n_sbs']}")
    print(f"  State   : {env.state_dim}")
    print(f"  Action  : {env.action_dim}")
    print(f"  Episodes: {train_cfg['episodes']}")
    print(f"{'='*60}\n")

    # Agent
    agent = DDPGAgent(
        state_dim   = env.state_dim,
        action_dim  = env.action_dim,
        lr_actor    = train_cfg['lr_actor'],
        lr_critic   = train_cfg['lr_critic'],
        gamma       = train_cfg['gamma'],
        tau         = train_cfg['tau'],
        buffer_size = train_cfg['buffer_size'],
        hidden_dim  = train_cfg['hidden_dim'],
    )

    # Metrics
    episode_rewards = []
    power_history   = []
    ee_history      = []
    best_reward     = -float('inf')

    for ep in tqdm(range(train_cfg['episodes']), desc='Training'):
        state = env.reset()
        ep_reward  = 0.0
        ep_powers  = []
        ep_ee      = []
        explore    = ep < train_cfg['episodes'] * train_cfg['explore_fraction']

        for step in range(train_cfg['max_steps']):
            action              = agent.select_action(state, explore=explore)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step(train_cfg['batch_size'])

            state      = next_state
            ep_reward += reward
            ep_powers.append(info['P_Net'])
            ep_ee.append(info['EE'])

            if done:
                break

        # Decay exploration noise
        agent.decay_noise(
            factor    = train_cfg['noise_decay'],
            min_scale = train_cfg['noise_std_min']
        )

        episode_rewards.append(ep_reward)
        power_history.append(np.mean(ep_powers))
        ee_history.append(np.mean(ep_ee))

        # Save best model
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(output_dir, 'best_model.pth'))

        # Periodic logging
        if (ep + 1) % train_cfg['log_interval'] == 0:
            avg_r   = np.mean(episode_rewards[-50:])
            avg_p   = np.mean(power_history[-50:])
            avg_ee  = np.mean(ee_history[-50:])
            print(f"\nEp {ep+1:5d} | "
                  f"Reward {avg_r:8.4f} | "
                  f"Power {avg_p:8.1f} W | "
                  f"EE {avg_ee:.2e} b/J | "
                  f"Noise {agent.noise_scale:.3f}")

        # Periodic checkpoint
        if (ep + 1) % train_cfg['save_interval'] == 0:
            agent.save(os.path.join(output_dir, f'model_ep{ep+1}.pth'))

    # Final save & plots
    agent.save(os.path.join(output_dir, 'final_model.pth'))
    plot_training(episode_rewards, power_history, ee_history, output_dir)

    np.save(os.path.join(output_dir, 'rewards.npy'),
            np.array(episode_rewards))
    np.save(os.path.join(output_dir, 'powers.npy'),
            np.array(power_history))
    np.save(os.path.join(output_dir, 'ee.npy'),
            np.array(ee_history))

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Reward     : {best_reward:.4f}")
    print(f"  Min Avg Power   : {min(power_history):.2f} W")
    print(f"  Max Avg EE      : {max(ee_history):.2e} bits/Joule")
    print(f"  Results saved → {output_dir}")
    print(f"{'='*60}\n")

    return agent


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    args   = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.n_ue:     config['network']['n_ue']           = args.n_ue
    if args.n_sbs:    config['network']['n_sbs']          = args.n_sbs
    if args.episodes: config['training']['episodes']      = args.episodes

    train(
        cfg        = config['network'],
        reward_cfg = config['reward'],
        train_cfg  = config['training'],
        output_dir = args.output,
        seed       = args.seed
    )
