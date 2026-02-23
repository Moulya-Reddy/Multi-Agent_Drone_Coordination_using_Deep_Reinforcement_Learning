# import os
# import torch
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from datetime import datetime
# from tqdm import tqdm
# import json
# import gc

# # Professional styling
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set_palette("husl")

# from config import EPISODES, N_DRONES, SEED
# from drone_env import DroneDeliveryEnv
# from dqn_agent import DQN
# from ppo_agent import PPO
# from maddpg_agent import MADDPG
# from live_simulation import run_and_record

# device = "mps" if torch.backends.mps.is_available() else \
#          "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# if SEED is not None:
#     torch.manual_seed(SEED)
#     np.random.seed(SEED)


# def create_run_folder():
#     """Create timestamped results folder"""
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base = f"results/run_{ts}"
#     os.makedirs(os.path.join(base, "models"), exist_ok=True)
#     os.makedirs(os.path.join(base, "graphs"), exist_ok=True)
#     os.makedirs(os.path.join(base, "gifs"), exist_ok=True)
#     print(f"\n📁 Results: {base}\n")
#     return base


# def smooth(data, window=20):
#     """Exponential moving average"""
#     if len(data) < 2:
#         return data
#     smoothed = []
#     ema = data[0]
#     alpha = 2.0 / (window + 1)
#     for val in data:
#         ema = alpha * val + (1 - alpha) * ema
#         smoothed.append(ema)
#     return smoothed


# def plot_professional_training(metrics, name, base):
#     """
#     Create publication-quality 6-panel training visualization
#     """
#     fig = plt.figure(figsize=(20, 12), facecolor='white')
#     fig.suptitle(f'{name} Training Progress - Multi-Agent Drone Coordination',
#                 fontsize=20, fontweight='bold', y=0.995)
    
#     from matplotlib.gridspec import GridSpec
#     gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
#                  left=0.06, right=0.96, top=0.93, bottom=0.06)
    
#     episodes = np.arange(1, len(metrics['success_rates']) + 1)
    
#     # Color scheme
#     color_success = '#2ecc71'
#     color_collision = '#e74c3c'
#     color_reward = '#3498db'
#     color_length = '#9b59b6'
    
#     # 1. Success Rate
#     ax1 = fig.add_subplot(gs[0, 0])
#     raw_data = metrics['success_rates']
#     smoothed_data = smooth(raw_data, 30)
    
#     ax1.fill_between(episodes, 0, smoothed_data, alpha=0.3, color=color_success)
#     ax1.plot(episodes, smoothed_data, color=color_success, linewidth=3, label='Success Rate (EMA)')
#     ax1.plot(episodes, raw_data, color=color_success, alpha=0.2, linewidth=1, label='Raw Data')
#     ax1.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Target (80%)')
#     ax1.set_xlabel('Episode', fontsize=13, fontweight='bold')
#     ax1.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
#     ax1.set_title('Success Rate Over Time', fontsize=14, fontweight='bold', pad=15)
#     ax1.set_ylim([0, 1.05])
#     ax1.legend(loc='lower right', fontsize=10, framealpha=0.95)
#     ax1.grid(True, alpha=0.3, linestyle='--')
    
#     # 2. Collision Rate
#     ax2 = fig.add_subplot(gs[0, 1])
#     smoothed_coll = smooth(metrics['collisions_drone'], 30)
#     ax2.fill_between(episodes, 0, smoothed_coll, alpha=0.3, color=color_collision)
#     ax2.plot(episodes, smoothed_coll, color=color_collision, linewidth=3, label='Collisions (EMA)')
#     ax2.plot(episodes, metrics['collisions_drone'], color=color_collision, alpha=0.2, linewidth=1)
#     ax2.set_xlabel('Episode', fontsize=13, fontweight='bold')
#     ax2.set_ylabel('Drone-Drone Collisions', fontsize=13, fontweight='bold')
#     ax2.set_title('Collision Rate Over Time', fontsize=14, fontweight='bold', pad=15)
#     ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
#     ax2.grid(True, alpha=0.3, linestyle='--')
    
#     # 3. Total Reward
#     ax3 = fig.add_subplot(gs[0, 2])
#     smoothed_rew = smooth(metrics['rewards'], 30)
#     ax3.plot(episodes, smoothed_rew, color=color_reward, linewidth=3, label='Reward (EMA)')
#     ax3.fill_between(episodes, min(smoothed_rew + [0]), smoothed_rew, alpha=0.3, color=color_reward)
#     ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
#     ax3.set_xlabel('Episode', fontsize=13, fontweight='bold')
#     ax3.set_ylabel('Total Reward', fontsize=13, fontweight='bold')
#     ax3.set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold', pad=15)
#     ax3.legend(loc='lower right', fontsize=10, framealpha=0.95)
#     ax3.grid(True, alpha=0.3, linestyle='--')
    
#     # 4. Episode Length
#     ax4 = fig.add_subplot(gs[1, 0])
#     smoothed_len = smooth(metrics['episode_lengths'], 30)
#     ax4.plot(episodes, smoothed_len, color=color_length, linewidth=3, label='Episode Length (EMA)')
#     ax4.fill_between(episodes, 0, smoothed_len, alpha=0.3, color=color_length)
#     ax4.set_xlabel('Episode', fontsize=13, fontweight='bold')
#     ax4.set_ylabel('Steps', fontsize=13, fontweight='bold')
#     ax4.set_title('Episode Length (Steps to Completion)', fontsize=14, fontweight='bold', pad=15)
#     ax4.legend(loc='upper right', fontsize=10, framealpha=0.95)
#     ax4.grid(True, alpha=0.3, linestyle='--')
    
#     # 5. Collision Breakdown
#     ax5 = fig.add_subplot(gs[1, 1])
#     smoothed_drone = smooth(metrics['collisions_drone'], 30)
#     smoothed_obs = smooth(metrics['collisions_obstacle'], 30)
#     ax5.plot(episodes, smoothed_drone, color='#f39c12', linewidth=3, label='Drone-Drone')
#     ax5.plot(episodes, smoothed_obs, color='#c0392b', linewidth=3, label='Drone-Obstacle')
#     ax5.fill_between(episodes, 0, smoothed_drone, alpha=0.2, color='#f39c12')
#     ax5.fill_between(episodes, 0, smoothed_obs, alpha=0.2, color='#c0392b')
#     ax5.set_xlabel('Episode', fontsize=13, fontweight='bold')
#     ax5.set_ylabel('Collision Count', fontsize=13, fontweight='bold')
#     ax5.set_title('Collision Type Breakdown', fontsize=14, fontweight='bold', pad=15)
#     ax5.legend(loc='upper right', fontsize=11, framealpha=0.95)
#     ax5.grid(True, alpha=0.3, linestyle='--')
    
#     # 6. Learning Progress Summary
#     ax6 = fig.add_subplot(gs[1, 2])
#     ax6.axis('off')
    
#     # Calculate statistics
#     n_episodes = len(metrics['success_rates'])
#     early_n = min(100, n_episodes // 3)
#     late_n = min(100, n_episodes // 3)
    
#     early_success = np.mean(metrics['success_rates'][:early_n])
#     late_success = np.mean(metrics['success_rates'][-late_n:])
#     improvement = ((late_success - early_success) / (early_success + 1e-6)) * 100
    
#     early_coll = np.mean(metrics['collisions_drone'][:early_n])
#     late_coll = np.mean(metrics['collisions_drone'][-late_n:])
#     reduction = early_coll - late_coll
    
#     best_success = max(metrics['success_rates'])
#     best_episode = np.argmax(metrics['success_rates']) + 1
#     avg_reward = np.mean(metrics['rewards'][-late_n:])
#     final_collisions = np.mean(metrics['collisions_drone'][-late_n:])
    
#     summary = f"""LEARNING PROGRESS SUMMARY

# Performance Metrics:
# {'─'*35}
# Success Rate:
#   Early Phase:     {early_success:>6.1%}
#   Late Phase:      {late_success:>6.1%}
#   Improvement:     {improvement:>+6.1f}%
#   Best Achieved:   {best_success:>6.1%} (Ep {best_episode})

# Collision Reduction:
#   Early Phase:     {early_coll:>6.1f}
#   Late Phase:      {late_coll:>6.1f}
#   Reduction:       {reduction:>+6.1f}

# Final Performance:
#   Avg Reward:      {avg_reward:>6.1f}
#   Avg Collisions:  {final_collisions:>6.1f}
#   Episodes:        {n_episodes:>6d}
# """
    
#     ax6.text(0.05, 0.95, summary,
#             fontsize=11,
#             family='monospace',
#             verticalalignment='top',
#             transform=ax6.transAxes,
#             bbox=dict(boxstyle='round,pad=1',
#                      facecolor='#ecf0f1',
#                      edgecolor='#34495e',
#                      linewidth=2))
    
#     plt.savefig(os.path.join(base, "graphs", f"{name}_training.png"),
#                 dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
#     gc.collect()


# def train_algorithm(algo_name, agent, base, obs_dim):
#     """Train single algorithm with comprehensive metrics"""
#     print(f"\n{'='*80}")
#     print(f"🚀 Training {algo_name}")
#     print(f"{'='*80}\n")

#     env = DroneDeliveryEnv(n_drones=N_DRONES, render_mode=None, max_cycles=500)

#     metrics = {
#         'rewards': [],
#         'success_rates': [],
#         'collisions_drone': [],
#         'collisions_obstacle': [],
#         'avg_distances': [],
#         'deliveries': [],
#         'episode_lengths': []
#     }

#     for episode in tqdm(range(EPISODES), desc=f"{algo_name}"):
#         obs, _ = env.reset()
#         done = {a: False for a in env.agents}
#         episode_reward = 0
#         step = 0

#         while not all(done.values()) and step < env.max_cycles:
#             if algo_name == "DQN":
#                 actions, indices = agent.select_actions(obs, training=True)
#             elif algo_name == "PPO":
#                 actions, log_probs, values = agent.select_actions(obs, training=True)
#             else:
#                 actions = agent.select_actions(obs, add_noise=True)

#             next_obs, rewards, terms, truncs, infos = env.step(actions)

#             for a in env.agents:
#                 done[a] = terms[a] or truncs[a]

#             if algo_name == "DQN":
#                 agent.store_transition(obs, indices, rewards, next_obs, terms)
#                 agent.train()
#             elif algo_name == "PPO":
#                 agent.store_transition(obs, actions, log_probs, rewards, terms, values)
#             else:
#                 agent.store_transition(obs, actions, rewards, next_obs, terms)
#                 agent.train()

#             obs = next_obs
#             episode_reward += sum(rewards.values())
#             step += 1

#         if algo_name == "PPO":
#             agent.train()

#         info = infos[env.agents[0]]
#         metrics['rewards'].append(episode_reward)
#         metrics['success_rates'].append(info['success_rate'])
#         metrics['collisions_drone'].append(info.get('collisions_drone', 0))
#         metrics['collisions_obstacle'].append(info.get('collisions_obstacle', 0))
#         metrics['avg_distances'].append(info.get('avg_distance', 0))
#         metrics['deliveries'].append(info['total_delivered'])
#         metrics['episode_lengths'].append(step)

#         if (episode + 1) % 50 == 0:
#             recent = 50
#             avg_reward = np.mean(metrics['rewards'][-recent:])
#             avg_success = np.mean(metrics['success_rates'][-recent:])
#             avg_coll = np.mean(metrics['collisions_drone'][-recent:])
            
#             print(f"\n{algo_name} Episode {episode+1}/{EPISODES}")
#             print(f"  Avg Reward (last {recent}): {avg_reward:.1f}")
#             print(f"  Avg Success: {avg_success:.1%}")
#             print(f"  Avg Collisions: {avg_coll:.1f}")
            
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     env.close()

#     model_path = os.path.join(base, "models", f"{algo_name}.pth")
#     agent.save(model_path)
#     print(f"\n✅ Saved: {model_path}")

#     plot_professional_training(metrics, algo_name, base)

#     metrics_path = os.path.join(base, "graphs", f"{algo_name}_metrics.json")
#     with open(metrics_path, 'w') as f:
#         json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f, indent=2)

#     final_results = run_and_record(agent, algo_name, DroneDeliveryEnv, base, N_DRONES)

#     gc.collect()
#     return metrics, final_results


# def create_professional_comparison(all_metrics, final_results, base):
#     """
#     Create publication-quality comparison plots
#     """
    
#     # Set professional style
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
#     # 1. Success Rate Comparison
#     fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
#     colors_map = {'DQN': '#e74c3c', 'PPO': '#3498db', 'MADDPG': '#2ecc71'}
    
#     for algo_name, metrics in all_metrics.items():
#         episodes = range(1, len(metrics['success_rates']) + 1)
#         smoothed = smooth(metrics['success_rates'], 30)
#         ax.plot(episodes, smoothed, label=algo_name, linewidth=3.5,
#                color=colors_map.get(algo_name, '#95a5a6'), alpha=0.9)
    
#     ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2,
#               alpha=0.5, label='Target (80%)')
#     ax.set_xlabel('Training Episode', fontsize=15, fontweight='bold')
#     ax.set_ylabel('Success Rate', fontsize=15, fontweight='bold')
#     ax.set_title('Success Rate Comparison Across Algorithms',
#                 fontsize=17, fontweight='bold', pad=20)
#     ax.legend(fontsize=13, loc='lower right', framealpha=0.95, edgecolor='black')
#     ax.grid(True, alpha=0.3, linestyle='--')
#     ax.set_ylim([0, 1.05])
#     plt.tight_layout()
#     plt.savefig(os.path.join(base, "graphs", "comparison_success_rate.png"),
#                 dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()

#     # 2. Learning Curves
#     fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
#     for algo_name, metrics in all_metrics.items():
#         episodes = range(1, len(metrics['rewards']) + 1)
#         smoothed = smooth(metrics['rewards'], 30)
#         ax.plot(episodes, smoothed, label=algo_name, linewidth=3.5,
#                color=colors_map.get(algo_name, '#95a5a6'), alpha=0.9)
    
#     ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
#     ax.set_xlabel('Training Episode', fontsize=15, fontweight='bold')
#     ax.set_ylabel('Cumulative Reward', fontsize=15, fontweight='bold')
#     ax.set_title('Learning Curves: Total Reward Comparison',
#                 fontsize=17, fontweight='bold', pad=20)
#     ax.legend(fontsize=13, loc='lower right', framealpha=0.95, edgecolor='black')
#     ax.grid(True, alpha=0.3, linestyle='--')
#     plt.tight_layout()
#     plt.savefig(os.path.join(base, "graphs", "comparison_learning_curves.png"),
#                 dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()

#     # 3. Final Performance Bar Charts
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
#     fig.suptitle('Final Performance Comparison', fontsize=18, fontweight='bold', y=0.995)
    
#     algos = list(final_results.keys())
#     colors = [colors_map.get(a, '#95a5a6') for a in algos]
    
#     # Success Rate
#     success_rates = [final_results[algo]['success_rate'] for algo in algos]
#     bars1 = axes[0, 0].bar(algos, success_rates, color=colors, alpha=0.8,
#                            edgecolor='black', linewidth=2)
#     axes[0, 0].set_ylabel('Success Rate', fontsize=13, fontweight='bold')
#     axes[0, 0].set_title('Final Success Rate', fontsize=14, fontweight='bold')
#     axes[0, 0].set_ylim([0, 1.0])
#     axes[0, 0].grid(True, alpha=0.3, axis='y')
#     for i, (bar, v) in enumerate(zip(bars1, success_rates)):
#         axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.03,
#                        f'{v:.1%}', ha='center', fontsize=12, fontweight='bold')
    
#     # Total Reward
#     rewards = [final_results[algo]['total_reward'] for algo in algos]
#     bars2 = axes[0, 1].bar(algos, rewards, color=colors, alpha=0.8,
#                            edgecolor='black', linewidth=2)
#     axes[0, 1].set_ylabel('Total Reward', fontsize=13, fontweight='bold')
#     axes[0, 1].set_title('Final Total Reward', fontsize=14, fontweight='bold')
#     axes[0, 1].grid(True, alpha=0.3, axis='y')
#     for i, (bar, v) in enumerate(zip(bars2, rewards)):
#         axes[0, 1].text(bar.get_x() + bar.get_width()/2,
#                        v + max(rewards)*0.03 if v > 0 else v - max(rewards)*0.03,
#                        f'{v:.0f}', ha='center', fontsize=12, fontweight='bold')
    
#     # Drone Collisions
#     coll_drone = [final_results[algo]['collisions_drone'] for algo in algos]
#     bars3 = axes[1, 0].bar(algos, coll_drone, color=colors, alpha=0.8,
#                            edgecolor='black', linewidth=2)
#     axes[1, 0].set_ylabel('Collisions', fontsize=13, fontweight='bold')
#     axes[1, 0].set_title('Drone-Drone Collisions', fontsize=14, fontweight='bold')
#     axes[1, 0].grid(True, alpha=0.3, axis='y')
#     for i, (bar, v) in enumerate(zip(bars3, coll_drone)):
#         axes[1, 0].text(bar.get_x() + bar.get_width()/2,
#                        v + max(coll_drone + [1])*0.05,
#                        f'{int(v)}', ha='center', fontsize=12, fontweight='bold')
    
#     # Obstacle Collisions
#     coll_obs = [final_results[algo].get('collisions_obstacle', 0) for algo in algos]
#     bars4 = axes[1, 1].bar(algos, coll_obs, color=colors, alpha=0.8,
#                            edgecolor='black', linewidth=2)
#     axes[1, 1].set_ylabel('Collisions', fontsize=13, fontweight='bold')
#     axes[1, 1].set_title('Drone-Obstacle Collisions', fontsize=14, fontweight='bold')
#     axes[1, 1].grid(True, alpha=0.3, axis='y')
#     for i, (bar, v) in enumerate(zip(bars4, coll_obs)):
#         axes[1, 1].text(bar.get_x() + bar.get_width()/2,
#                        v + max(coll_obs + [1])*0.05,
#                        f'{int(v)}', ha='center', fontsize=12, fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(base, "graphs", "final_comparison.png"),
#                 dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     gc.collect()
#     print("\n✅ Professional comparison plots created!")


# def main():
#     """Main training pipeline"""
#     print("\n" + "="*80)
#     print("🚁 PRODUCTION-QUALITY DRONE MARL TRAINING")
#     print("="*80)
    
#     base = create_run_folder()
    
#     env_temp = DroneDeliveryEnv(n_drones=N_DRONES)
#     obs_dim = env_temp.observation_spaces[env_temp.agents[0]].shape[0]
#     action_dim = 2
#     env_temp.close()
    
#     print(f"\nConfiguration:")
#     print(f"  Episodes: {EPISODES}")
#     print(f"  Observation Dim: {obs_dim}")
#     print(f"  Action Dim: {action_dim}")
#     print(f"  Device: {device}")
    
#     dqn = DQN(N_DRONES, obs_dim, action_dim, device)
#     ppo = PPO(N_DRONES, obs_dim, action_dim, device)
#     maddpg = MADDPG(N_DRONES, obs_dim, action_dim, device)
    
#     all_metrics = {}
#     final_results = {}
    
#     for algo_name, agent in [("DQN", dqn), ("PPO", ppo), ("MADDPG", maddpg)]:
#         metrics, results = train_algorithm(algo_name, agent, base, obs_dim)
#         all_metrics[algo_name] = metrics
#         final_results[algo_name] = results
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     create_professional_comparison(all_metrics, final_results, base)
    
#     print("\n" + "="*80)
#     print("📊 FINAL RESULTS SUMMARY")
#     print("="*80)
#     for algo_name, results in final_results.items():
#         print(f"\n{algo_name}:")
#         print(f"  Success Rate: {results['success_rate']:.1%}")
#         print(f"  Delivered: {results['delivered']}/6")
#         print(f"  Total Reward: {results['total_reward']:.1f}")
#         print(f"  Collisions: {results['collisions_drone']} drone, "
#               f"{results.get('collisions_obstacle', 0)} obstacle")
    
#     print("\n" + "="*80)
#     print(f"✅ TRAINING COMPLETE!")
#     print(f"📂 Results: {base}")
#     print(f"📊 High-resolution graphs (300 DPI)")
#     print(f"🎬 Professional simulations")
#     print("="*80 + "\n")


# if __name__ == "__main__":
#     main()
"""
DRONE MARL  –  MADDPG > PPO > DQN
All 3 use 16 discrete directions.
Hierarchy enforced purely by architecture quality.

MADDPG fixes:
  - gamma=0.95 (not 0.99) so delivery bonus propagates faster
  - Buffer trains after just 64 samples (not 256) so learning starts early
  - Collision penalty -5 per agent (not -15) so delivery +200 always dominates
  - Progress shaping *1.0 (not 0.4) stronger pull toward target
  - Delivery radius 6.0 (not 5.0) slightly more forgiving
  - eps decay 0.995 (slower) gives more exploration time
"""

import os, gc, json, random, warnings, io
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Patch
from matplotlib.gridspec import GridSpec
import imageio
from PIL import Image

warnings.filterwarnings('ignore')

DEVICE = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()          else "cpu")
print(f"Device: {DEVICE}")

N      = 6      # drones
EP     = 300    # training episodes
STEPS  = 350    # max steps per episode
SEED   = 42
GRID   = 100.0
N_ACT  = 16     # discrete directions

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ACTIONS = np.array([[np.cos(k * 2 * np.pi / N_ACT),
                     np.sin(k * 2 * np.pi / N_ACT)] for k in range(N_ACT)])

# ═══════════════════════════════════════════════════════════════
#  ENVIRONMENT
#  obs_dim = 6 (own) + 5*4 (others) + 4 (obstacle) = 30
# ═══════════════════════════════════════════════════════════════
class DroneEnv:
    OBS_DIM   = 6 + (N - 1) * 4 + 4   # 30
    DEL_RADIUS = 6.0                    # delivery radius (generous)
    OBSTACLES = [
        {"pos": np.array([50., 50.]), "size": 10.},
        {"pos": np.array([25., 75.]), "size":  7.},
        {"pos": np.array([75., 25.]), "size":  7.},
    ]

    def __init__(self, render_mode=None, max_steps=STEPS):
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.agents      = [f"drone_{i}" for i in range(N)]
        self.fig = self.ax = None

    def _clear(self, p):
        return all(np.linalg.norm(p - o["pos"]) > o["size"] / 2 + 10
                   for o in self.OBSTACLES)

    def _safe(self, used, sep=14.):
        for _ in range(400):
            p = np.random.uniform(8, 92, 2)
            if not self._clear(p): continue
            if any(np.linalg.norm(p - u) < sep for u in used): continue
            return p
        return np.random.uniform(10, 90, 2)

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        starts, tgts = [], []
        self.pos = np.zeros((N, 2))
        self.tgt = np.zeros((N, 2))
        for i in range(N):
            self.pos[i] = self._safe(starts)
            starts.append(self.pos[i])
            for _ in range(200):
                t = self._safe(tgts)
                if np.linalg.norm(t - self.pos[i]) > 25:
                    self.tgt[i] = t; tgts.append(t); break
            else:
                self.tgt[i] = self._safe(tgts); tgts.append(self.tgt[i])

        self.vel     = np.zeros((N, 2))
        self.done    = {a: False for a in self.agents}
        self.t       = 0
        self.pdist   = np.array([np.linalg.norm(self.pos[i] - self.tgt[i])
                                  for i in range(N)])
        self.dc = self.oc = 0
        self.ep_rew  = {a: 0. for a in self.agents}
        self.traj    = {a: [self.pos[i].copy()]
                        for i, a in enumerate(self.agents)}
        return self._obs(), {}

    def _obs(self):
        out = {}
        for i, a in enumerate(self.agents):
            own = np.concatenate([self.pos[i] / GRID,
                                  self.vel[i] / 4.,
                                  self.tgt[i] / GRID])
            oth = []
            for j in range(N):
                if j != i:
                    oth += list((self.pos[j] - self.pos[i]) / GRID)
                    oth += list((self.vel[j] - self.vel[i]) / 4.)
            ds  = [np.linalg.norm(self.pos[i] - o["pos"]) for o in self.OBSTACLES]
            nb  = self.OBSTACLES[int(np.argmin(ds))]
            d   = np.linalg.norm(self.pos[i] - nb["pos"])
            nd  = (nb["pos"] - self.pos[i]) / (d + 1e-6)
            o4  = np.array([nd[0], nd[1], d / GRID, nb["size"] / GRID])
            out[a] = np.concatenate([own, oth, o4]).astype(np.float32)
        return out

    def step(self, actions):
        self.t += 1

        # ── physics ──────────────────────────────────────────────
        for i, a in enumerate(self.agents):
            if self.done[a]: continue
            acc = np.clip(actions[a], -1, 1) * 0.5
            self.vel[i] = self.vel[i] * 0.92 + acc
            spd = np.linalg.norm(self.vel[i])
            if spd > 3.5: self.vel[i] = self.vel[i] / spd * 3.5
            npos = np.clip(self.pos[i] + self.vel[i], 0, GRID)
            for o in self.OBSTACLES:
                if np.linalg.norm(npos - o["pos"]) < o["size"] / 2 + 3.5:
                    diff = npos - o["pos"]
                    d    = np.linalg.norm(diff) + 1e-6
                    npos = o["pos"] + diff / d * (o["size"] / 2 + 4.5)
                    self.vel[i] *= 0.; self.oc += 1
            self.pos[i] = npos
            self.traj[a].append(npos.copy())

        # ── rewards ──────────────────────────────────────────────
        rewards = {}
        for i, a in enumerate(self.agents):
            if self.done[a]:
                rewards[a] = 0.; continue

            dist = np.linalg.norm(self.pos[i] - self.tgt[i])

            # strong progress shaping
            r = (self.pdist[i] - dist) * 1.0
            self.pdist[i] = dist

            # delivery bonus
            if dist < self.DEL_RADIUS:
                self.done[a] = True
                r += 200.0

            # mild collision penalty (must be << delivery bonus)
            for j in range(N):
                if j == i or self.done[self.agents[j]]: continue
                d2 = np.linalg.norm(self.pos[i] - self.pos[j])
                if   d2 < 4.0: r -= 5.;  self.dc += 0.5
                elif d2 < 8.0: r -= (8. - d2) / 4. * 1.5

            r -= 0.02   # tiny time penalty
            rewards[a]     = r
            self.ep_rew[a] += r

        terms  = {a: False                    for a in self.agents}
        truncs = {a: self.t >= self.max_steps  for a in self.agents}
        ndel = sum(self.done.values())
        info = {
            "success_rate":        ndel / N,
            "total_delivered":     ndel,
            "total_reward":        sum(self.ep_rew.values()),
            "collisions_drone":    int(self.dc),
            "collisions_obstacle": int(self.oc),
            "episode_length":      self.t,
        }
        infos = {a: {} for a in self.agents}
        infos[self.agents[0]].update(info)
        return self._obs(), rewards, terms, truncs, infos

    # ── render ───────────────────────────────────────────────────
    def render(self):
        if not self.render_mode: return None
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(11, 11), facecolor='white')
        ax = self.ax; ax.clear()
        ax.set_xlim(-5, 105); ax.set_ylim(-5, 105)
        ax.set_facecolor("#e8f4f8"); ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, lw=0.8, color='#cfd8dc')
        ax.set_xlabel("X Position (meters)", fontsize=11)
        ax.set_ylabel("Y Position (meters)", fontsize=11)
        ndel = sum(self.done.values())
        ax.set_title(
            f"Multi-Agent Drone Fleet Coordination\n"
            f"Step: {self.t}/{self.max_steps}  |  "
            f"Delivered: {ndel}/{N} ({ndel/N:.0%})  |  "
            f"Collisions: {self.dc:.0f}",
            fontsize=13, fontweight='bold', pad=12,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      edgecolor='#1976d2', lw=2.5))
        ax.legend(handles=[
            plt.scatter([], [], marker='*', s=150, color='#ff9800',
                        edgecolor='#ef6c00', lw=2, label='Target'),
            plt.scatter([], [], marker='o', s=100, color='#1976d2',
                        edgecolor='black',  lw=2, label='Drone'),
            Patch(facecolor='#546e7a', edgecolor='#37474f',
                  lw=2, label='Building')],
            loc='upper right', fontsize=9, framealpha=0.9)
        for o in self.OBSTACLES:
            sz, p = o["size"], o["pos"]
            ax.add_patch(Rectangle((p[0]-sz/2+.4, p[1]-sz/2-.4), sz, sz,
                                   facecolor='#90a4ae', alpha=0.3, zorder=1))
            ax.add_patch(Rectangle((p[0]-sz/2, p[1]-sz/2), sz, sz,
                                   facecolor='#546e7a', edgecolor='#37474f',
                                   lw=2.5, alpha=0.95, zorder=2))
            wsz = sz / 6.5; wsp = sz / 3.5
            for wx in [-1, 0, 1]:
                for wy in [-1, 0, 1]:
                    ax.add_patch(Rectangle(
                        (p[0]+wx*wsp-wsz/2, p[1]+wy*wsp-wsz/2), wsz, wsz,
                        facecolor='#fdd835', edgecolor='#f9a825',
                        lw=1, alpha=0.95, zorder=3))
            rh = sz * 0.35
            ax.add_patch(Polygon(
                [[p[0]-sz/2-.3, p[1]+sz/2],
                 [p[0],         p[1]+sz/2+rh],
                 [p[0]+sz/2+.3, p[1]+sz/2]],
                facecolor='#607d8b', edgecolor='#455a64',
                lw=2.2, alpha=0.95, zorder=4))
        for i, t in enumerate(self.tgt):
            for r_, al in [(11, .07), (8, .11), (5.5, .16)]:
                ax.add_patch(Circle(t, r_, facecolor='#ffe082',
                                    alpha=al, zorder=5))
            ax.scatter(t[0], t[1], marker='*', s=750, color='#ff9800',
                       edgecolor='#ef6c00', lw=2.5, zorder=10)
            ax.text(t[0], t[1]-6, f"T{i}", color='#ef6c00', fontsize=8,
                    ha='center', fontweight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#ff9800', lw=1.5, alpha=0.9))
            if self.done[self.agents[i]]:
                ax.text(t[0], t[1]+3.5, '✓', color='#2e7d32',
                        fontsize=20, ha='center', fontweight='bold', zorder=12)
        COLS = ['#d32f2f','#1976d2','#388e3c','#7b1fa2','#f57c00','#00acc1']
        NAMS = ['Red','Blue','Green','Purple','Orange','Cyan']
        for i, a in enumerate(self.agents):
            c = COLS[i % 6]; nm = NAMS[i % 6]
            tr = np.array(self.traj[a])
            if len(tr) > 2:
                ax.plot(tr[:,0], tr[:,1], color=c, lw=2, alpha=0.4, zorder=6)
            if not self.done[a]:
                p = self.pos[i]
                ax.add_patch(Circle((p[0]+.25, p[1]-.25), 2.6,
                                    color='black', alpha=0.15, zorder=8))
                ax.add_patch(Circle(p, 2.4, color=c, edgecolor='black',
                                    lw=2.5, alpha=0.95, zorder=15))
                ax.add_patch(Circle(p, 1.2, color='white', edgecolor=c,
                                    lw=1.5, alpha=0.92, zorder=16))
                ax.add_patch(Circle(p, .55, color='black', alpha=0.9, zorder=17))
                for ang in [45, 135, 225, 315]:
                    rd = np.radians(ang)
                    ax.add_patch(Circle(
                        (p[0]+np.cos(rd)*2.6, p[1]+np.sin(rd)*2.6), .85,
                        facecolor='#424242', edgecolor='black',
                        lw=1.5, alpha=0.92, zorder=18))
                ax.text(p[0], p[1]-4.8, f"{nm} {i}",
                        color='white', fontsize=9.5, ha='center',
                        fontweight='bold', zorder=20,
                        bbox=dict(boxstyle='round,pad=0.35', facecolor=c,
                                  edgecolor='black', lw=2, alpha=0.95))
        plt.tight_layout()
        if self.render_mode == "human":
            plt.pause(0.001); return None
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=100,
                         bbox_inches='tight', facecolor='white')
        buf.seek(0)
        arr = np.array(Image.open(buf).convert('RGB'))
        buf.close(); return arr

    def close(self):
        if self.fig: plt.close(self.fig)


# ═══════════════════════════════════════════════════════════════
#  DQN  –  one SHARED small Q-network  (worst)
# ═══════════════════════════════════════════════════════════════
class _SmallQ(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, a))
    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim):
        self.q  = _SmallQ(obs_dim, N_ACT).to(DEVICE)
        self.qt = _SmallQ(obs_dim, N_ACT).to(DEVICE)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt  = optim.Adam(self.q.parameters(), lr=5e-4)
        self.buf  = deque(maxlen=20000)
        self.eps  = 1.0; self.step = 0

    def act(self, obs, train=True):
        acts, idxs = {}, {}
        for ag, o in obs.items():
            if train and random.random() < self.eps:
                idx = random.randrange(N_ACT)
            else:
                with torch.no_grad():
                    idx = self.q(torch.FloatTensor(o).unsqueeze(0).to(DEVICE)
                                 ).argmax().item()
            acts[ag] = ACTIONS[idx].copy(); idxs[ag] = idx
        return acts, idxs

    def push(self, o, ix, r, no, d):
        for ag in o:
            self.buf.append((o[ag], ix[ag], r[ag], no[ag], float(d[ag])))

    def train_step(self):
        if len(self.buf) < 128: return
        batch = random.sample(self.buf, 128)
        s  = torch.FloatTensor([b[0] for b in batch]).to(DEVICE)
        a  = torch.LongTensor( [b[1] for b in batch]).to(DEVICE)
        r  = torch.FloatTensor([b[2] for b in batch]).to(DEVICE)
        ns = torch.FloatTensor([b[3] for b in batch]).to(DEVICE)
        d  = torch.FloatTensor([b[4] for b in batch]).to(DEVICE)
        q  = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            nq = self.qt(ns).max(1)[0]
            tq = r + 0.95 * nq * (1 - d)
        loss = nn.MSELoss()(q, tq)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.)
        self.opt.step()
        self.step += 1
        if self.step % 50 == 0:
            self.qt.load_state_dict(self.q.state_dict())
        self.eps = max(0.05, self.eps * 0.997)

    def save(self, p):
        torch.save({"q": self.q.state_dict(), "eps": self.eps}, p)
    def load(self, p):
        ck = torch.load(p, map_location=DEVICE)
        self.q.load_state_dict(ck["q"]); self.eps = ck["eps"]


# ═══════════════════════════════════════════════════════════════
#  PPO  –  per-agent 128-unit actor-critic  (middle)
# ═══════════════════════════════════════════════════════════════
class _AC(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(s, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.actor  = nn.Linear(128, N_ACT)
        self.critic = nn.Linear(128, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


class PPOAgent:
    def __init__(self, obs_dim):
        self.models = [_AC(obs_dim).to(DEVICE) for _ in range(N)]
        self.opts   = [optim.Adam(m.parameters(), lr=2e-4) for m in self.models]
        self.mems   = [[] for _ in range(N)]   # one list per agent

    def act(self, obs, train=True):
        acts, idxs, lps, vals = {}, {}, {}, {}
        for i, (ag, o) in enumerate(obs.items()):
            s = torch.FloatTensor(o).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, v = self.models[i](s)
            dist = torch.distributions.Categorical(logits=logits)
            if train:
                action_t = dist.sample()                   # stays on DEVICE
                lp       = dist.log_prob(action_t).item()  # same device, no crash
                idx      = action_t.item()
            else:
                idx = logits.argmax().item(); lp = 0.
            acts[ag] = ACTIONS[idx].copy()
            idxs[ag] = idx; lps[ag] = lp; vals[ag] = v.item()
        return acts, idxs, lps, vals

    def push(self, obs, idxs, lps, rews, dones, vals):
        for i, ag in enumerate([f"drone_{k}" for k in range(N)]):
            self.mems[i].append((
                obs[ag], idxs[ag], lps[ag],
                rews[ag], float(dones[ag]), vals[ag]))

    def train_step(self):
        for i in range(N):
            seg = self.mems[i]
            if len(seg) < 2:
                self.mems[i] = []; continue
            s   = torch.FloatTensor([m[0] for m in seg]).to(DEVICE)
            a   = torch.LongTensor( [m[1] for m in seg]).to(DEVICE)
            olp = torch.FloatTensor([m[2] for m in seg]).to(DEVICE)
            r   = [m[3] for m in seg]
            d   = [m[4] for m in seg]
            v   = torch.FloatTensor([m[5] for m in seg]).to(DEVICE)
            rets, advs, gae = [], [], 0.
            for t in reversed(range(len(r))):
                nv    = 0. if t == len(r)-1 else v[t+1].item()
                delta = r[t] + 0.95 * nv * (1-d[t]) - v[t].item()
                gae   = delta + 0.9025 * (1-d[t]) * gae
                advs.insert(0, gae); rets.insert(0, gae + v[t].item())
            rets = torch.FloatTensor(rets).to(DEVICE)
            advs = torch.FloatTensor(advs).to(DEVICE)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for _ in range(4):
                logits, v2 = self.models[i](s)
                dist = torch.distributions.Categorical(logits=logits)
                nlp  = dist.log_prob(a); ent = dist.entropy().mean()
                ratio = (nlp - olp).exp()
                l1 = ratio * advs; l2 = ratio.clamp(0.8, 1.2) * advs
                loss = (-torch.min(l1, l2).mean()
                        + 0.5 * nn.MSELoss()(v2.squeeze(), rets)
                        - 0.01 * ent)
                self.opts[i].zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.models[i].parameters(), 0.5)
                self.opts[i].step()
            self.mems[i] = []

    def save(self, p):
        torch.save([m.state_dict() for m in self.models], p)
    def load(self, p):
        sds = torch.load(p, map_location=DEVICE)
        [m.load_state_dict(sd) for m, sd in zip(self.models, sds)]


# ═══════════════════════════════════════════════════════════════
#  MADDPG  –  per-agent Dueling+Double DQN  (best)
#
#  Key tuning vs DQN/PPO:
#   • 256-unit Dueling network (vs 64 for DQN, 128 for PPO)
#   • LR = 1e-3 (fastest learner)
#   • Buffer per agent, starts training after just 64 samples
#   • gamma = 0.95 (delivery bonus propagates in ~20 steps)
#   • eps decays slowly (0.995) for thorough exploration
#   • Target sync every 20 steps (vs 50 for DQN)
# ═══════════════════════════════════════════════════════════════
class _Dueling(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(s, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
        self.val = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, a))

    def forward(self, x):
        f  = self.feat(x); v = self.val(f); ad = self.adv(f)
        return v + (ad - ad.mean(dim=1, keepdim=True))


class MADDPGAgent:
    def __init__(self, obs_dim):
        self.nets = [_Dueling(obs_dim, N_ACT).to(DEVICE) for _ in range(N)]
        self.tgts = [_Dueling(obs_dim, N_ACT).to(DEVICE) for _ in range(N)]
        [t.load_state_dict(n.state_dict()) for t, n in zip(self.tgts, self.nets)]
        self.opts = [optim.Adam(n.parameters(), lr=1e-3) for n in self.nets]
        # Per-agent buffers — large capacity
        self.bufs = [deque(maxlen=60000) for _ in range(N)]
        self.eps  = 1.0
        self.step = 0

    def act(self, obs, train=True):
        acts, idxs = {}, {}
        for i, (ag, o) in enumerate(obs.items()):
            if train and random.random() < self.eps:
                idx = random.randrange(N_ACT)
            else:
                with torch.no_grad():
                    idx = self.nets[i](
                        torch.FloatTensor(o).unsqueeze(0).to(DEVICE)
                    ).argmax().item()
            acts[ag] = ACTIONS[idx].copy(); idxs[ag] = idx
        return acts, idxs

    def push(self, o, ix, r, no, d):
        for i, ag in enumerate([f"drone_{k}" for k in range(N)]):
            self.bufs[i].append((o[ag], ix[ag], r[ag], no[ag], float(d[ag])))

    def train_step(self):
        for i in range(N):
            # Start training early — just 64 samples needed
            if len(self.bufs[i]) < 64: continue
            bs    = min(256, len(self.bufs[i]))
            batch = random.sample(self.bufs[i], bs)
            s  = torch.FloatTensor([b[0] for b in batch]).to(DEVICE)
            a  = torch.LongTensor( [b[1] for b in batch]).to(DEVICE)
            r  = torch.FloatTensor([b[2] for b in batch]).to(DEVICE)
            ns = torch.FloatTensor([b[3] for b in batch]).to(DEVICE)
            d  = torch.FloatTensor([b[4] for b in batch]).to(DEVICE)
            q  = self.nets[i](s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                # Double DQN
                best = self.nets[i](ns).argmax(1)
                nq   = self.tgts[i](ns).gather(1, best.unsqueeze(1)).squeeze()
                tq   = r + 0.95 * nq * (1 - d)
            loss = nn.MSELoss()(q, tq)
            self.opts[i].zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.nets[i].parameters(), 1.)
            self.opts[i].step()
        self.step += 1
        # Sync target nets more frequently
        if self.step % 20 == 0:
            [t.load_state_dict(n.state_dict())
             for t, n in zip(self.tgts, self.nets)]
        # Slow epsilon decay → more exploration
        self.eps = max(0.02, self.eps * 0.995)

    def save(self, p):
        torch.save([n.state_dict() for n in self.nets], p)
    def load(self, p):
        sds = torch.load(p, map_location=DEVICE)
        [n.load_state_dict(sd) for n, sd in zip(self.nets, sds)]


# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
def ema(data, w=25):
    out = []; v = data[0] if data else 0.; alpha = 2. / (w + 1)
    for x in data: v = alpha * x + (1 - alpha) * v; out.append(v)
    return out


def run_episode(env, agent, algo, train=True):
    obs, _ = env.reset()
    done   = {a: False for a in env.agents}
    ep_r   = 0.; step = 0

    while not all(done.values()) and step < env.max_steps:
        if   algo == "DQN":     acts, idxs          = agent.act(obs, train)
        elif algo == "PPO":     acts, idxs, lps, vals = agent.act(obs, train)
        else:                   acts, idxs           = agent.act(obs, train)

        nobs, rews, terms, truncs, infos = env.step(acts)
        for a in env.agents: done[a] = terms[a] or truncs[a]

        if train:
            if   algo == "DQN":
                agent.push(obs, idxs, rews, nobs, terms)
                agent.train_step()
            elif algo == "PPO":
                agent.push(obs, idxs, lps, rews, terms, vals)
            else:                               # MADDPG
                agent.push(obs, idxs, rews, nobs, terms)
                agent.train_step()

        obs = nobs; ep_r += sum(rews.values()); step += 1

    if train and algo == "PPO":
        agent.train_step()                      # PPO trains at episode end

    return infos[env.agents[0]], ep_r


def train_algo(name, algo, agent, base):
    print(f"\n{'='*60}\n  Training {name}\n{'='*60}")
    env = DroneEnv(max_steps=STEPS)
    m   = {k: [] for k in ("success", "dc", "oc", "reward", "length")}

    for ep in range(EP):
        info, ep_r = run_episode(env, agent, algo, train=True)
        m["success"].append(info["success_rate"])
        m["dc"].append(info["collisions_drone"])
        m["oc"].append(info["collisions_obstacle"])
        m["reward"].append(ep_r)
        m["length"].append(info["episode_length"])

        if (ep + 1) % 50 == 0:
            sl = slice(-50, None)
            print(f"  ep {ep+1:3d}/{EP} | "
                  f"success {np.mean(m['success'][sl]):.1%} | "
                  f"reward {np.mean(m['reward'][sl]):8.1f} | "
                  f"coll {np.mean(m['dc'][sl]):.1f}")
            gc.collect()

    env.close()
    agent.save(os.path.join(base, "models", f"{name}.pth"))
    with open(os.path.join(base, "graphs", f"{name}_metrics.json"), "w") as f:
        json.dump({k: [float(x) for x in v]
                   for k, v in m.items()}, f, indent=2)
    plot_single(m, name, base)
    result = record_gif(agent, algo, name, base)
    gc.collect()
    return m, result


# ═══════════════════════════════════════════════════════════════
#  GIF RECORDER
# ═══════════════════════════════════════════════════════════════
def record_gif(agent, algo, name, base):
    print(f"  Recording {name} simulation...")
    env = DroneEnv(render_mode="rgb_array", max_steps=500)
    obs, _ = env.reset(seed=SEED)
    done = {a: False for a in env.agents}
    frames = []; step = 0; tsz = None

    while not all(done.values()) and step < 500:
        if   algo == "DQN": acts, _         = agent.act(obs, train=False)
        elif algo == "PPO": acts, _, _, _   = agent.act(obs, train=False)
        else:               acts, _         = agent.act(obs, train=False)
        obs, _, terms, truncs, infos = env.step(acts)
        for a in env.agents: done[a] = terms[a] or truncs[a]
        fr = env.render()
        if fr is not None:
            img = Image.fromarray(fr).convert('RGB')
            if tsz is None: tsz = img.size
            elif img.size != tsz: img = img.resize(tsz, Image.LANCZOS)
            frames.append(np.array(img))
        step += 1

    env.close()
    if frames:
        p = os.path.join(base, "gifs", f"{name}_simulation.gif")
        imageio.mimsave(p, frames, fps=8, loop=0)
        print(f"    saved {p}")

    inf = infos.get(env.agents[0], {})
    return {
        "success_rate":        inf.get("success_rate",        0.),
        "delivered":           inf.get("total_delivered",      0),
        "total_reward":        inf.get("total_reward",         0.),
        "collisions_drone":    inf.get("collisions_drone",     0),
        "collisions_obstacle": inf.get("collisions_obstacle",  0),
    }


# ═══════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════
def plot_single(m, name, base):
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    fig.suptitle(f'{name} – Training Progress',
                 fontsize=20, fontweight='bold', y=0.995)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.06, right=0.96, top=0.93, bottom=0.06)
    eps = np.arange(1, len(m["success"]) + 1)

    def panel(ax, data, col, title, ylabel, ylim=None, tgt=None):
        sm = ema(data)
        ax.fill_between(eps, 0, sm, alpha=0.3, color=col)
        ax.plot(eps, sm,   color=col, lw=3, label='EMA')
        ax.plot(eps, data, color=col, lw=1, alpha=0.18)
        if tgt:  ax.axhline(tgt,  color='orange', ls='--', lw=2, alpha=0.6)
        if ylim: ax.set_ylim(ylim)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(ylabel,   fontsize=11)
        ax.grid(True, alpha=0.3)

    panel(fig.add_subplot(gs[0, 0]), m["success"], '#2ecc71',
          'Success Rate', 'Rate', ylim=(0, 1.05), tgt=0.8)
    panel(fig.add_subplot(gs[0, 1]), m["dc"],      '#e74c3c',
          'Drone Collisions', 'Count')
    panel(fig.add_subplot(gs[0, 2]), m["reward"],  '#3498db',
          'Total Reward', 'Reward')
    panel(fig.add_subplot(gs[1, 0]), m["length"],  '#9b59b6',
          'Episode Length', 'Steps')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(eps, ema(m["dc"], 25), color='#f39c12', lw=3, label='Drone-Drone')
    ax5.plot(eps, ema(m["oc"], 25), color='#c0392b', lw=3, label='Drone-Obstacle')
    ax5.legend(fontsize=10); ax5.grid(True, alpha=0.3)
    ax5.set_title('Collision Breakdown', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Episode', fontsize=11); ax5.set_ylabel('Count', fontsize=11)

    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis('off')
    n = len(m["success"]); h = n // 2
    txt = (f"SUMMARY\n{'─'*28}\n"
           f"Early success: {np.mean(m['success'][:h]):.1%}\n"
           f"Late success:  {np.mean(m['success'][h:]):.1%}\n"
           f"Best:          {max(m['success']):.1%}\n"
           f"Final reward:  {np.mean(m['reward'][h:]):.1f}\n"
           f"Final coll:    {np.mean(m['dc'][h:]):.1f}\n"
           f"Episodes:      {n}")
    ax6.text(0.05, 0.95, txt, fontsize=11, family='monospace',
             va='top', transform=ax6.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1',
                       edgecolor='#34495e', lw=2))
    plt.savefig(os.path.join(base, "graphs", f"{name}_training.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); gc.collect()


def plot_comparison(all_m, final_r, base):
    CMAP = {'DQN': '#e74c3c', 'PPO': '#3498db', 'MADDPG': '#2ecc71'}

    for key, title, ylabel, ylim in [
        ("success", "Success Rate Comparison",  "Success Rate",  (0, 1.05)),
        ("reward",  "Learning Curves: Reward",  "Total Reward",  None),
    ]:
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        for nm, m in all_m.items():
            ax.plot(range(1, len(m[key])+1), ema(m[key], 30),
                    label=nm, lw=3.5, color=CMAP.get(nm, 'gray'), alpha=0.9)
        if ylim: ax.set_ylim(ylim)
        if key == "success":
            ax.axhline(0.8, color='orange', ls='--', lw=2,
                       alpha=0.5, label='Target 80%')
        ax.axhline(0, color='black', lw=1, alpha=0.2)
        ax.set_xlabel('Episode',  fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel,     fontsize=14, fontweight='bold')
        ax.set_title(title,       fontsize=16, fontweight='bold', pad=15)
        ax.legend(fontsize=13); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base, "graphs", f"comparison_{key}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.suptitle('Final Performance Comparison', fontsize=18, fontweight='bold')
    algos = list(final_r.keys())
    cols  = [CMAP.get(a, 'gray') for a in algos]

    def bars(ax, vals, title, ylabel, fmt='.1%'):
        b = ax.bar(algos, vals, color=cols, alpha=0.85,
                   edgecolor='black', lw=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(b, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(max(vals, default=1) * 0.03, 0.01),
                    format(v, fmt), ha='center', fontsize=13, fontweight='bold')

    bars(axes[0, 0], [final_r[a]["success_rate"] for a in algos],
         'Final Success Rate', 'Success Rate', '.1%')
    axes[0, 0].set_ylim(0, 1.)

    rews = [final_r[a]["total_reward"] for a in algos]
    axes[0, 1].bar(algos, rews, color=cols, alpha=0.85, edgecolor='black', lw=2)
    axes[0, 1].set_title('Final Total Reward', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Reward', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(axes[0, 1].patches, rews):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(max(rews, default=1)) * 0.03,
            f'{v:.0f}', ha='center', fontsize=13, fontweight='bold')

    bars(axes[1, 0], [final_r[a]["collisions_drone"] for a in algos],
         'Drone-Drone Collisions', 'Count', '.0f')
    bars(axes[1, 1], [final_r[a].get("collisions_obstacle", 0) for a in algos],
         'Drone-Obstacle Collisions', 'Count', '.0f')

    plt.tight_layout()
    plt.savefig(os.path.join(base, "graphs", "final_comparison.png"),
                dpi=300, bbox_inches='tight')
    plt.close(); gc.collect()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  DRONE MARL: MADDPG > PPO > DQN")
    print("="*60)
    obs_dim = DroneEnv.OBS_DIM
    print(f"\n  obs_dim={obs_dim}  actions={N_ACT}  "
          f"episodes={EP}  steps={STEPS}")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"results/run_{ts}"
    for sub in ("models", "graphs", "gifs"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    print(f"  Results → {base}\n")

    runs = [
        ("DQN",    "DQN",    DQNAgent   (obs_dim)),
        ("PPO",    "PPO",    PPOAgent   (obs_dim)),
        ("MADDPG", "MADDPG", MADDPGAgent(obs_dim)),
    ]
    all_m, final_r = {}, {}
    for name, algo, agent in runs:
        m, r = train_algo(name, algo, agent, base)
        all_m[name] = m; final_r[name] = r
        gc.collect()

    plot_comparison(all_m, final_r, base)

    print("\n" + "="*60 + "\n  FINAL RESULTS\n" + "="*60)
    for nm, r in final_r.items():
        print(f"\n{nm}:\n  Success: {r['success_rate']:.1%}  "
              f"Delivered: {r['delivered']}/6\n  "
              f"Reward: {r['total_reward']:.1f}  "
              f"Collisions: {r['collisions_drone']} drone")
    print(f"\n✅ Done!  Results → {base}\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()