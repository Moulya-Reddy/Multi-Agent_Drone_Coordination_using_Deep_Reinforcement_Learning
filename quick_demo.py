# """
# QUICK DEMO SCRIPT FOR PANEL PRESENTATION
# Run trained models with custom drone/target positions
# Fast simulation for live demonstration
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import argparse
# import os

# from environment.drone_env import DroneDeliveryEnv
# from agents.dqn_agent import DQN
# from agents.ppo_agent import PPO
# from agents.maddpg_agent import MADDPG


# class QuickDemo:
#     """Fast demo for trained models"""
    
#     def __init__(self, model_path, algorithm, n_drones=6):
#         self.n_drones = n_drones
#         self.algorithm = algorithm
        
#         # Load environment
#         self.env = DroneDeliveryEnv(
#             n_drones=n_drones, 
#             render_mode=None,
#             max_cycles=500  # Shorter for demo
#         )
        
#         # Get dimensions
#         obs_dim = self.env.observation_spaces[self.env.agents[0]].shape[0]
#         action_dim = 2
        
#         # Load agent
#         device = "mps" if torch.backends.mps.is_available() else \
#                  "cuda" if torch.cuda.is_available() else "cpu"
        
#         if algorithm == "DQN":
#             self.agent = DQN(n_drones, obs_dim, action_dim, device)
#         elif algorithm == "PPO":
#             self.agent = PPO(n_drones, obs_dim, action_dim, device)
#         elif algorithm == "MADDPG":
#             self.agent = MADDPG(n_drones, obs_dim, action_dim, device)
#         else:
#             raise ValueError(f"Unknown algorithm: {algorithm}")
        
#         # Load trained weights
#         if os.path.exists(model_path):
#             if algorithm == "DQN":
#                 checkpoint = torch.load(model_path, map_location=device)
#                 self.agent.q_net.load_state_dict(checkpoint['q_net'])
#                 print(f"✅ Loaded {algorithm} model from {model_path}")
#             elif algorithm == "PPO":
#                 checkpoint = torch.load(model_path, map_location=device)
#                 self.agent.model.load_state_dict(checkpoint['model'])
#                 print(f"✅ Loaded {algorithm} model from {model_path}")
#             elif algorithm == "MADDPG":
#                 checkpoint = torch.load(model_path, map_location=device)
#                 for i, actor in enumerate(self.agent.actors):
#                     actor.load_state_dict(checkpoint['actors'][i])
#                 self.agent.critic.load_state_dict(checkpoint['critic'])
#                 print(f"✅ Loaded {algorithm} model from {model_path}")
#         else:
#             print(f"⚠️  Model not found: {model_path}")
#             print(f"   Using untrained {algorithm} model")
    
#     def set_custom_scenario(self, start_positions=None, target_positions=None):
#         """Set custom starting and target positions"""
#         if start_positions is not None:
#             self.env.positions = np.array(start_positions)
        
#         if target_positions is not None:
#             self.env.targets = np.array(target_positions)
        
#         self.env.velocities = np.zeros((self.n_drones, 2))
#         self.env.delivered = {a: False for a in self.env.agents}
#         self.env.timestep = 0
#         self.env.trajectories = {
#             a: [self.env.positions[i].copy()] 
#             for i, a in enumerate(self.env.agents)
#         }
    
#     def run_episode(self, max_steps=500, render=False):
#         """Run one episode and return metrics"""
#         obs, _ = self.env.reset()
#         done = {a: False for a in self.env.agents}
#         step = 0
        
#         total_reward = 0
#         frames = []
        
#         while not all(done.values()) and step < max_steps:
#             # Get actions
#             if self.algorithm == "DQN":
#                 actions, _ = self.agent.select_actions(obs, training=False)
#             elif self.algorithm == "PPO":
#                 actions, _, _ = self.agent.select_actions(obs, training=False)
#             else:  # MADDPG
#                 actions = self.agent.select_actions(obs, add_noise=False)
            
#             obs, rewards, terms, truncs, infos = self.env.step(actions)
            
#             for a in self.env.agents:
#                 done[a] = terms[a] or truncs[a]
            
#             total_reward += sum(rewards.values())
#             step += 1
            
#             # Early stop if all delivered
#             if infos[self.env.agents[0]]['total_delivered'] == self.n_drones:
#                 break
        
#         # Get final metrics
#         info = infos[self.env.agents[0]]
        
#         return {
#             'delivered': info['total_delivered'],
#             'success_rate': info['success_rate'],
#             'total_reward': info['total_reward'],
#             'collisions_drone': info.get('collisions_drone', 0),
#             'collisions_obstacle': info.get('collisions_obstacle', 0),
#             'steps': step
#         }
    
#     def live_visualization(self, max_steps=500):
#         """Live matplotlib visualization for panel demo"""
#         obs, _ = self.env.reset()
#         done = {a: False for a in self.env.agents}
        
#         # Setup plot
#         fig, ax = plt.subplots(figsize=(10, 10))
#         plt.ion()
        
#         step = 0
#         while not all(done.values()) and step < max_steps:
#             # Get actions
#             if self.algorithm == "DQN":
#                 actions, _ = self.agent.select_actions(obs, training=False)
#             elif self.algorithm == "PPO":
#                 actions, _, _ = self.agent.select_actions(obs, training=False)
#             else:
#                 actions = self.agent.select_actions(obs, add_noise=False)
            
#             obs, rewards, terms, truncs, infos = self.env.step(actions)
            
#             for a in self.env.agents:
#                 done[a] = terms[a] or truncs[a]
            
#             # Visualize every 5 steps
#             if step % 5 == 0:
#                 ax.clear()
#                 self._draw_state(ax)
#                 plt.pause(0.01)
            
#             step += 1
            
#             if infos[self.env.agents[0]]['total_delivered'] == self.n_drones:
#                 # Show final state for 2 seconds
#                 ax.clear()
#                 self._draw_state(ax)
#                 plt.pause(2.0)
#                 break
        
#         plt.ioff()
#         plt.close()
        
#         return infos[self.env.agents[0]]
    
#     def _draw_state(self, ax):
#         """Draw current state"""
#         ax.set_xlim(-5, 105)
#         ax.set_ylim(-5, 105)
#         ax.set_facecolor("#e8f4f8")
#         ax.set_title(f"{self.algorithm} Demo | Step: {self.env.timestep} | "
#                     f"Delivered: {sum(self.env.delivered.values())}/{self.n_drones}",
#                     fontsize=14, fontweight='bold')
#         ax.grid(True, alpha=0.3)
        
#         # Draw obstacles
#         from matplotlib.patches import Rectangle
#         for obs in self.env.obstacles:
#             size = obs['size']
#             pos = obs['pos']
#             rect = Rectangle(
#                 (pos[0] - size/2, pos[1] - size/2),
#                 size, size,
#                 facecolor='#34495e',
#                 alpha=0.7
#             )
#             ax.add_patch(rect)
        
#         # Draw targets
#         for i, t in enumerate(self.env.targets):
#             ax.scatter(t[0], t[1], marker="*", s=500, 
#                       color="gold", edgecolor="orange", linewidth=2)
#             if self.env.delivered[self.env.agents[i]]:
#                 ax.text(t[0], t[1]+4, "✓", color="green", 
#                        fontsize=20, ha="center", fontweight='bold')
        
#         # Draw drones
#         colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#e67e22"]
#         for i, a in enumerate(self.env.agents):
#             if not self.env.delivered[a]:
#                 pos = self.env.positions[i]
#                 color = colors[i % len(colors)]
                
#                 # Trajectory
#                 traj = np.array(self.env.trajectories[a])
#                 if len(traj) > 1:
#                     ax.plot(traj[:, 0], traj[:, 1], 
#                            color=color, linewidth=2, alpha=0.6)
                
#                 # Drone
#                 ax.scatter(pos[0], pos[1], s=300, color=color, 
#                           edgecolor='black', linewidth=2)
#                 ax.text(pos[0], pos[1]-3, f"D{i}", 
#                        color="white", fontsize=10, 
#                        ha="center", fontweight='bold')


# # ============================================================================
# # PRESET SCENARIOS FOR PANEL DEMO
# # ============================================================================

# SCENARIOS = {
#     "easy": {
#         "name": "Easy Scenario - Short Distances",
#         "starts": [
#             [20, 20], [30, 20], [40, 20],
#             [20, 30], [30, 30], [40, 30]
#         ],
#         "targets": [
#             [20, 70], [30, 70], [40, 70],
#             [20, 80], [30, 80], [40, 80]
#         ]
#     },
    
#     "medium": {
#         "name": "Medium Scenario - Around Obstacles",
#         "starts": [
#             [15, 15], [85, 15], [15, 85],
#             [85, 85], [30, 15], [70, 85]
#         ],
#         "targets": [
#             [85, 85], [15, 85], [85, 15],
#             [15, 15], [70, 15], [30, 85]
#         ]
#     },
    
#     "hard": {
#         "name": "Hard Scenario - Cross Formation",
#         "starts": [
#             [15, 50], [85, 50], [50, 15],
#             [50, 85], [25, 25], [75, 75]
#         ],
#         "targets": [
#             [85, 50], [15, 50], [50, 85],
#             [50, 15], [75, 75], [25, 25]
#         ]
#     },
    
#     "convergence": {
#         "name": "Convergence Test - All to Center",
#         "starts": [
#             [15, 15], [85, 15], [15, 85],
#             [85, 85], [15, 50], [85, 50]
#         ],
#         "targets": [
#             [45, 45], [55, 45], [45, 55],
#             [55, 55], [40, 50], [60, 50]
#         ]
#     },
    
#     "spread": {
#         "name": "Spread Test - Center to Corners",
#         "starts": [
#             [45, 45], [55, 45], [45, 55],
#             [55, 55], [50, 40], [50, 60]
#         ],
#         "targets": [
#             [15, 15], [85, 15], [15, 85],
#             [85, 85], [15, 50], [85, 50]
#         ]
#     }
# }


# def main():
#     parser = argparse.ArgumentParser(description="Quick Demo for Panel Presentation")
#     parser.add_argument("--model", type=str, required=True,
#                        help="Path to trained model (.pth file)")
#     parser.add_argument("--algorithm", type=str, required=True,
#                        choices=["DQN", "PPO", "MADDPG"],
#                        help="Algorithm type")
#     parser.add_argument("--scenario", type=str, default="medium",
#                        choices=list(SCENARIOS.keys()),
#                        help="Preset scenario to test")
#     parser.add_argument("--live", action="store_true",
#                        help="Show live visualization")
#     parser.add_argument("--runs", type=int, default=1,
#                        help="Number of test runs")
    
#     args = parser.parse_args()
    
#     print("\n" + "="*70)
#     print("🚁 QUICK MODEL DEMO FOR PANEL PRESENTATION")
#     print("="*70)
    
#     # Load model
#     demo = QuickDemo(args.model, args.algorithm)
    
#     # Get scenario
#     scenario = SCENARIOS[args.scenario]
#     print(f"\n📋 Testing: {scenario['name']}")
#     print(f"   Algorithm: {args.algorithm}")
#     print(f"   Runs: {args.runs}")
    
#     # Run tests
#     results = []
#     for run in range(args.runs):
#         print(f"\n🔄 Run {run+1}/{args.runs}...")
        
#         # Reset with custom positions
#         demo.env.reset()
#         demo.set_custom_scenario(
#             start_positions=scenario['starts'],
#             target_positions=scenario['targets']
#         )
        
#         if args.live and run == 0:  # Show live only for first run
#             print("   🎬 Live visualization (close window to continue)...")
#             result = demo.live_visualization(max_steps=500)
#             result = {
#                 'delivered': result['total_delivered'],
#                 'success_rate': result['success_rate'],
#                 'total_reward': result['total_reward'],
#                 'collisions_drone': result.get('collisions_drone', 0),
#                 'collisions_obstacle': result.get('collisions_obstacle', 0),
#                 'steps': result.get('episode_length', 0)
#             }
#         else:
#             result = demo.run_episode(max_steps=500)
        
#         results.append(result)
        
#         print(f"   ✅ Delivered: {result['delivered']}/6")
#         print(f"   📊 Success Rate: {result['success_rate']:.1%}")
#         print(f"   🎯 Reward: {result['total_reward']:.1f}")
#         print(f"   ⚡ Steps: {result['steps']}")
#         print(f"   💥 Collisions: {result['collisions_drone']} drone, "
#               f"{result['collisions_obstacle']} obstacle")
    
#     # Summary
#     print("\n" + "="*70)
#     print("📊 SUMMARY")
#     print("="*70)
    
#     avg_success = np.mean([r['success_rate'] for r in results])
#     avg_reward = np.mean([r['total_reward'] for r in results])
#     avg_steps = np.mean([r['steps'] for r in results])
#     avg_collisions = np.mean([r['collisions_drone'] + r['collisions_obstacle'] 
#                               for r in results])
    
#     print(f"\nAverage across {args.runs} runs:")
#     print(f"  Success Rate: {avg_success:.1%}")
#     print(f"  Reward: {avg_reward:.1f}")
#     print(f"  Steps: {avg_steps:.0f}")
#     print(f"  Collisions: {avg_collisions:.1f}")
    
#     # Pass/Fail criteria
#     print(f"\n{'='*70}")
#     print("✅ EVALUATION")
#     print("="*70)
    
#     if avg_success >= 0.80:
#         print("✅ EXCELLENT - Success rate ≥ 80%")
#     elif avg_success >= 0.60:
#         print("✅ GOOD - Success rate ≥ 60%")
#     elif avg_success >= 0.40:
#         print("⚠️  FAIR - Success rate ≥ 40%")
#     else:
#         print("❌ NEEDS IMPROVEMENT - Success rate < 40%")
    
#     if avg_collisions < 5:
#         print("✅ LOW COLLISIONS - Very safe navigation")
#     elif avg_collisions < 10:
#         print("✅ ACCEPTABLE COLLISIONS - Safe enough")
#     else:
#         print("⚠️  HIGH COLLISIONS - Needs better obstacle avoidance")
    
#     print("\n" + "="*70 + "\n")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Demo script for models trained with main.py
Loads saved MADDPG/PPO/DQN models and runs live visualization
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse

# Configuration matching main.py training
N_DRONES  = 6
N_ZONES   = 3
ZONE_CAP  = 2
GRID      = 100.0
N_ACT     = 8
OBS_DIM   = 43  # From main.py: 6 + 25 + 12
JOINT_DIM = 258  # 43 * 6

DEVICE = "cpu"

DIRS = np.array([[np.cos(k*2*np.pi/N_ACT), np.sin(k*2*np.pi/N_ACT)]
                 for k in range(N_ACT)], dtype=np.float32)
ZONES = np.array([[20., 20.], [50., 80.], [80., 20.]], dtype=np.float32)
OBSTACLES = [(np.array([35., 50.]), 7.),
             (np.array([65., 50.]), 7.),
             (np.array([50., 35.]), 5.)]


# ═══════════════════════════════════════════════════════════════
# Copy agent architectures from main.py
# ═══════════════════════════════════════════════════════════════
import torch.nn as nn

class DQNAgent:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),
                                      nn.Linear(128,128),nn.ReLU(),
                                      nn.Linear(128,N_ACT))
        def forward(self,x): return self.net(x)

    def __init__(self):
        self.q = self.Net().to(DEVICE)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"✅ Loaded DQN model")

    def act(self, obs_dict):
        acts = {}
        for ag, o in obs_dict.items():
            with torch.no_grad():
                t = torch.FloatTensor(o).unsqueeze(0).to(DEVICE)
                idx = self.q(t).argmax().item()
            acts[ag] = idx
        return acts


class PPOAgent:
    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.sh=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),
                                  nn.Linear(128,128),nn.ReLU())
            self.ac=nn.Linear(128,N_ACT); self.cr=nn.Linear(128,1)
        def forward(self,x): h=self.sh(x); return self.ac(h),self.cr(h)

    def __init__(self):
        self.nets = nn.ModuleList([self.AC().to(DEVICE) for _ in range(N_DRONES)])

    def load(self, path):
        state_dicts = torch.load(path, map_location=DEVICE)
        for net, sd in zip(self.nets, state_dicts):
            net.load_state_dict(sd)
        print(f"✅ Loaded PPO model")

    def act(self, obs_dict):
        acts = {}
        for i, ag in enumerate([f"d{j}" for j in range(N_DRONES)]):
            with torch.no_grad():
                s = torch.FloatTensor(obs_dict[ag]).unsqueeze(0).to(DEVICE)
                logits, _ = self.nets[i](s)
                idx = logits.argmax().item()
            acts[ag] = idx
        return acts


class MADDPGAgent:
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ReLU(),
                                   nn.Linear(128,128),nn.ReLU(),
                                   nn.Linear(128,N_ACT))
        def forward(self,x): return self.net(x)

    def __init__(self):
        self.actors = nn.ModuleList([self.Actor().to(DEVICE) for _ in range(N_DRONES)])

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        print(f"✅ Loaded MADDPG model")

    def act(self, obs_dict):
        acts = {}
        for i in range(N_DRONES):
            ag = f"d{i}"
            with torch.no_grad():
                t = torch.FloatTensor(obs_dict[ag]).unsqueeze(0).to(DEVICE)
                idx = self.actors[i](t).argmax().item()
            acts[ag] = idx
        return acts


# ═══════════════════════════════════════════════════════════════
# Simple environment for demo (copy from main.py)
# ═══════════════════════════════════════════════════════════════
class DroneEnv:
    def __init__(self):
        self.agents = [f"d{i}" for i in range(N_DRONES)]

    def _clear(self, p, m=10.):
        for o,r in OBSTACLES:
            if np.linalg.norm(p-o) < r+m: return False
        return True

    def _spawn(self, used, sep=14.):
        for _ in range(600):
            p = np.random.uniform(5., 95., 2).astype(np.float32)
            if not self._clear(p): continue
            if all(np.linalg.norm(p-u) >= sep for u in used): return p
        return np.random.uniform(10., 90., 2).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.pos    = np.zeros((N_DRONES,2), dtype=np.float32)
        self.vel    = np.zeros((N_DRONES,2), dtype=np.float32)
        self.target = np.array([i // ZONE_CAP for i in range(N_DRONES)], dtype=int)
        spawned = []
        for i in range(N_DRONES):
            self.pos[i] = self._spawn(spawned + list(ZONES))
            spawned.append(self.pos[i])
        self.done = np.zeros(N_DRONES, dtype=bool)
        self.t = 0
        return self._obs()

    def _obs(self):
        zc = np.zeros(N_ZONES, dtype=float)
        for i in range(N_DRONES):
            if not self.done[i]: zc[self.target[i]] += 1.
        out = {}
        for i,ag in enumerate(self.agents):
            tgt  = ZONES[self.target[i]]
            diff = tgt - self.pos[i]
            dist = np.linalg.norm(diff)
            tdir = diff / (dist + 1e-6)
            own  = np.array([self.pos[i,0]/GRID, self.pos[i,1]/GRID,
                              self.vel[i,0]/4., self.vel[i,1]/4.,
                              tdir[0], tdir[1]], dtype=np.float32)
            oth = []
            for j in range(N_DRONES):
                if j == i: continue
                rp = (self.pos[j]-self.pos[i])/GRID
                rv = (self.vel[j]-self.vel[i])/4.
                st = float(self.target[j] == self.target[i])
                oth += [rp[0], rp[1], rv[0], rv[1], st]
            zfeat = []
            for k in range(N_ZONES):
                rp   = (ZONES[k]-self.pos[i])/GRID
                d    = np.linalg.norm(ZONES[k]-self.pos[i])/GRID
                conf = float(zc[k] > ZONE_CAP)
                zfeat += [rp[0], rp[1], d, conf]
            out[ag] = np.concatenate([own, np.array(oth,dtype=np.float32),
                                       np.array(zfeat,dtype=np.float32)]).astype(np.float32)
        return out

    def step(self, act_dict):
        self.t += 1
        for i,ag in enumerate(self.agents):
            if self.done[i]: continue
            acc = DIRS[act_dict[ag]] * 0.6
            self.vel[i] = self.vel[i]*0.90 + acc
            spd = np.linalg.norm(self.vel[i])
            if spd > 4.0: self.vel[i] = self.vel[i]/spd*4.0
            npos = np.clip(self.pos[i]+self.vel[i], 0., GRID)
            for o,r in OBSTACLES:
                if np.linalg.norm(npos-o) < r+3.5:
                    diff = npos-o
                    npos = o + diff/(np.linalg.norm(diff)+1e-6)*(r+4.5)
                    self.vel[i] *= 0.
            self.pos[i] = npos
        
        for i in range(N_DRONES):
            if self.done[i]: continue
            z = self.target[i]
            dist = np.linalg.norm(self.pos[i]-ZONES[z])
            if dist < 8.:
                at_zone = [j for j in range(N_DRONES) 
                           if not self.done[j] and 
                           np.linalg.norm(self.pos[j]-ZONES[z]) < 8.]
                if len(at_zone) <= ZONE_CAP:
                    self.done[i] = True
        
        return self._obs()


# ═══════════════════════════════════════════════════════════════
# Live visualization
# ═══════════════════════════════════════════════════════════════
def run_demo(agent, algo_name, max_steps=500):
    env = DroneEnv()
    obs = env.reset(seed=42)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#1a1a1a')
    
    for step in range(max_steps):
        # Get actions
        acts = agent.act(obs)
        
        # Step environment
        obs = env.step(acts)
        
        # Visualize every 5 steps
        if step % 5 == 0 or all(env.done):
            ax.clear()
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.set_facecolor('#0d1117')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.1, color='white')
            ax.tick_params(colors='#666666')
            for sp in ax.spines.values(): sp.set_color('#333333')
            
            n_done = int(env.done.sum())
            ax.set_title(
                f"{algo_name} Demo | Step {step} | Delivered {n_done}/{N_DRONES}",
                color='white', fontsize=14, fontweight='bold', pad=10)
            
            # Obstacles
            for opos, orad in OBSTACLES:
                circle = Circle(opos, orad, facecolor='#30363d',
                               edgecolor='#6e7681', lw=2, zorder=3)
                ax.add_patch(circle)
            
            # Zones
            ZCOLS = ['#ff7b72', '#79c0ff', '#56d364']
            for k, (zpos, zc) in enumerate(zip(ZONES, ZCOLS)):
                for r, a in [(14, .05), (10, .09), (7, .14)]:
                    circle = Circle(zpos, r, facecolor=zc, alpha=a, zorder=2)
                    ax.add_patch(circle)
                circle = Circle(zpos, 7, facecolor='none', edgecolor=zc,
                               lw=2, ls='--', alpha=0.7, zorder=5)
                ax.add_patch(circle)
                ax.scatter(*zpos, marker='H', s=300, color=zc,
                          edgecolor='white', lw=1.5, zorder=6)
            
            # Drones
            DCOLS = ['#ff6e6e','#6eb5ff','#6effb3','#ffb86e','#e06eff','#fff06e']
            for i in range(N_DRONES):
                c = DCOLS[i]
                p = env.pos[i]
                
                if not env.done[i]:
                    circle = Circle(p, 2.6, color=c, edgecolor='white',
                                   lw=2, alpha=0.95, zorder=10)
                    ax.add_patch(circle)
                    circle = Circle(p, 1., color='white', alpha=0.85, zorder=11)
                    ax.add_patch(circle)
                    
                    for ang in [45, 135, 225, 315]:
                        rd = np.radians(ang)
                        rx = p[0] + np.cos(rd) * 3.1
                        ry = p[1] + np.sin(rd) * 3.1
                        circle = Circle((rx, ry), 1., facecolor='#21262d',
                                       edgecolor=c, lw=1.5, alpha=0.9, zorder=12)
                        ax.add_patch(circle)
                    
                    ax.text(p[0], p[1]-5.5, f"D{i}",
                           color='white', fontsize=8, ha='center',
                           fontweight='bold', zorder=15,
                           bbox=dict(boxstyle='round,pad=0.25',
                                    facecolor=c, edgecolor='white',
                                    lw=1, alpha=0.9))
                else:
                    tgt = ZONES[env.target[i]]
                    ax.text(tgt[0], tgt[1], '✓',
                           color='#56d364', fontsize=14,
                           ha='center', va='center',
                           fontweight='bold', zorder=20)
            
            plt.pause(0.1)
        
        if all(env.done):
            print(f"\n✅ All drones delivered in {step} steps!")
            plt.pause(2)
            break
    
    plt.ioff()
    plt.show()
    
    n_done = int(env.done.sum())
    print(f"\nFinal: {n_done}/{N_DRONES} delivered")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth model")
    parser.add_argument("--algo", required=True, choices=["DQN","PPO","MADDPG"])
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"  DEMO: {args.algo}")
    print("="*60 + "\n")
    
    # Load agent
    if args.algo == "DQN":
        agent = DQNAgent()
        agent.load(args.model)
    elif args.algo == "PPO":
        agent = PPOAgent()
        agent.load(args.model)
    else:
        agent = MADDPGAgent()
        agent.load(args.model)
    
    # Run
    run_demo(agent, args.algo)


if __name__ == "__main__":
    main()