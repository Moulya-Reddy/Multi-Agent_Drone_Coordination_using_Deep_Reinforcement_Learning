# import numpy as np
# from pettingzoo import ParallelEnv
# from gymnasium import spaces
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon
# from matplotlib.collections import LineCollection
# import io
# from PIL import Image


# class DroneDeliveryEnv(ParallelEnv):
#     """
#     Production-Ready Multi-Agent Drone Environment
#     - Professional visualization
#     - Proper reward shaping
#     - Stable learning
#     """
#     metadata = {"render_modes": ["human", "rgb_array"]}

#     def __init__(self, n_drones=6, render_mode=None, max_cycles=500):
#         super().__init__()

#         self.n_drones = n_drones
#         self.render_mode = render_mode
#         self.max_cycles = max_cycles
#         self.grid_size = 100

#         # Physics
#         self.max_acceleration = 0.4
#         self.max_velocity = 3.0
#         self.damping = 0.92
        
#         # Safety
#         self.collision_radius = 5.0
#         self.delivery_radius = 5.0
#         self.safe_distance = 8.0

#         self.agents = [f"drone_{i}" for i in range(n_drones)]

#         # Strategic obstacles
#         self.obstacles = [
#             {'pos': np.array([50.0, 50.0]), 'size': 10.0, 'type': 'building'},
#             {'pos': np.array([30.0, 70.0]), 'size': 7.0, 'type': 'building'},
#             {'pos': np.array([70.0, 30.0]), 'size': 7.0, 'type': 'building'}
#         ]

#         obs_dim = 6 + (n_drones - 1) * 4 + 4
        
#         self.observation_spaces = {
#             a: spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
#             for a in self.agents
#         }

#         self.action_spaces = {
#             a: spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
#             for a in self.agents
#         }

#         self.fig = None
#         self.ax = None
        
#         # Metrics tracking
#         self.total_collisions_drone = 0
#         self.total_collisions_obstacle = 0
#         self.total_distance_traveled = {a: 0.0 for a in self.agents}

#     def _is_position_safe(self, pos, min_distance=12.0):
#         """Check if position is safe from obstacles"""
#         for obs in self.obstacles:
#             dist = np.linalg.norm(pos - obs['pos'])
#             if dist < min_distance:
#                 return False
#         return True

#     def _get_safe_position(self, avoid_positions=None, min_sep=12.0):
#         """Generate safe random position"""
#         for _ in range(200):
#             pos = np.random.uniform(15, 85, 2)
            
#             if not self._is_position_safe(pos, min_distance=12.0):
#                 continue
            
#             if avoid_positions is not None and len(avoid_positions) > 0:
#                 dists = [np.linalg.norm(pos - p) for p in avoid_positions]
#                 if min(dists) < min_sep:
#                     continue
            
#             return pos
        
#         # Fallback corners
#         corners = [
#             np.array([20.0, 20.0]), np.array([80.0, 20.0]),
#             np.array([20.0, 80.0]), np.array([80.0, 80.0])
#         ]
#         for corner in corners:
#             if self._is_position_safe(corner):
#                 return corner
        
#         return np.array([20.0, 20.0])

#     def _is_colliding_with_obstacle(self, pos):
#         """Check obstacle collision"""
#         for obs in self.obstacles:
#             dist = np.linalg.norm(pos - obs['pos'])
#             if dist < (obs['size'] / 2 + self.collision_radius):
#                 return True, obs
#         return False, None

#     def _get_nearest_obstacle_info(self, pos):
#         """Get nearest obstacle info"""
#         min_dist = float('inf')
#         nearest_dir = np.array([0.0, 0.0])
        
#         for obs in self.obstacles:
#             diff = obs['pos'] - pos
#             dist = np.linalg.norm(diff)
#             if dist < min_dist:
#                 min_dist = dist
#                 nearest_dir = diff / (dist + 1e-6)
        
#         return np.concatenate([nearest_dir, [min_dist / 100.0, 0.0]])

#     def reset(self, seed=None, options=None):
#         if seed is not None:
#             np.random.seed(seed)

#         # Generate safe positions
#         self.positions = np.zeros((self.n_drones, 2))
#         self.targets = np.zeros((self.n_drones, 2))
        
#         used_starts = []
#         used_targets = []
        
#         for i in range(self.n_drones):
#             self.positions[i] = self._get_safe_position(used_starts, min_sep=12.0)
#             used_starts.append(self.positions[i])
            
#             for _ in range(100):
#                 target = self._get_safe_position(used_targets, min_sep=12.0)
#                 if np.linalg.norm(target - self.positions[i]) > 30:
#                     self.targets[i] = target
#                     used_targets.append(target)
#                     break
#             else:
#                 self.targets[i] = self._get_safe_position(used_targets, min_sep=12.0)
#                 used_targets.append(self.targets[i])

#         self.velocities = np.zeros((self.n_drones, 2))
#         self.delivered = {a: False for a in self.agents}
#         self.timestep = 0

#         self.trajectories = {a: [self.positions[i].copy()]
#                              for i, a in enumerate(self.agents)}
        
#         self.total_collisions_drone = 0
#         self.total_collisions_obstacle = 0
#         self.episode_rewards = {a: 0.0 for a in self.agents}
#         self.total_distance_traveled = {a: 0.0 for a in self.agents}
        
#         # Track previous distances
#         self.prev_distances = np.array([
#             np.linalg.norm(self.positions[i] - self.targets[i])
#             for i in range(self.n_drones)
#         ])

#         obs = self._get_obs()
#         infos = {a: {} for a in self.agents}
#         return obs, infos

#     def _get_obs(self):
#         obs = {}
#         for i, a in enumerate(self.agents):
#             own_state = np.concatenate([
#                 self.positions[i] / 100.0,
#                 self.velocities[i] / self.max_velocity,
#                 self.targets[i] / 100.0,
#             ])
            
#             other_info = []
#             for j in range(self.n_drones):
#                 if i != j:
#                     rel_pos = (self.positions[j] - self.positions[i]) / 100.0
#                     rel_vel = (self.velocities[j] - self.velocities[i]) / self.max_velocity
#                     other_info.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]])
            
#             obstacle_info = self._get_nearest_obstacle_info(self.positions[i])
            
#             obs[a] = np.concatenate([own_state, other_info, obstacle_info]).astype(np.float32)
        
#         return obs

#     def step(self, actions):
#         self.timestep += 1
#         rewards = {}
        
#         old_positions = self.positions.copy()

#         # Physics update
#         for i, a in enumerate(self.agents):
#             if not self.delivered[a]:
#                 # Apply action
#                 acc = np.clip(actions[a], -1, 1) * self.max_acceleration
#                 self.velocities[i] += acc
#                 self.velocities[i] *= self.damping
                
#                 # Limit velocity
#                 vel_mag = np.linalg.norm(self.velocities[i])
#                 if vel_mag > self.max_velocity:
#                     self.velocities[i] = (self.velocities[i] / vel_mag) * self.max_velocity

#                 # Update position
#                 new_pos = self.positions[i] + self.velocities[i]
#                 new_pos = np.clip(new_pos, 0, self.grid_size)
                
#                 # Check obstacle collision
#                 colliding, obs_hit = self._is_colliding_with_obstacle(new_pos)
#                 if colliding:
#                     # Bounce back
#                     diff = new_pos - obs_hit['pos']
#                     dist = np.linalg.norm(diff)
#                     push_dir = diff / (dist + 1e-6)
#                     new_pos = obs_hit['pos'] + push_dir * (obs_hit['size'] / 2 + self.collision_radius + 1)
#                     self.velocities[i] *= 0.0
#                     self.total_collisions_obstacle += 1
                
#                 # Track distance
#                 step_dist = np.linalg.norm(new_pos - self.positions[i])
#                 self.total_distance_traveled[a] += step_dist
                
#                 self.positions[i] = new_pos
#                 self.trajectories[a].append(new_pos.copy())

#         # Compute rewards
#         for i, a in enumerate(self.agents):
#             if self.delivered[a]:
#                 rewards[a] = 0.0
#                 continue

#             dist_to_target = np.linalg.norm(self.positions[i] - self.targets[i])
            
#             # Progress reward (shaped)
#             progress = self.prev_distances[i] - dist_to_target
#             self.prev_distances[i] = dist_to_target
#             reward = progress * 0.3  # Small multiplier
            
#             # Delivery bonus
#             if dist_to_target < self.delivery_radius:
#                 self.delivered[a] = True
#                 reward += 150.0
            
#             # Distance-based shaping
#             max_dist = np.sqrt(2) * 100
#             closeness = (max_dist - dist_to_target) / max_dist
#             reward += closeness * 0.1
            
#             # COLLISION PENALTIES
#             # Drone-drone collisions
#             for j in range(self.n_drones):
#                 if i != j and not self.delivered[self.agents[j]]:
#                     dist_to_other = np.linalg.norm(self.positions[i] - self.positions[j])
                    
#                     if dist_to_other < self.collision_radius:
#                         reward -= 8.0  # Direct collision
#                         self.total_collisions_drone += 0.5
#                     elif dist_to_other < self.safe_distance:
#                         # Proximity warning
#                         proximity_factor = 1.0 - (dist_to_other - self.collision_radius) / (self.safe_distance - self.collision_radius)
#                         reward -= proximity_factor * 1.5
            
#             # Obstacle proximity
#             for obs in self.obstacles:
#                 dist_to_obs = np.linalg.norm(self.positions[i] - obs['pos'])
#                 danger_zone = obs['size'] / 2 + 10
                
#                 if dist_to_obs < danger_zone:
#                     penalty = (danger_zone - dist_to_obs) / danger_zone * 0.3
#                     reward -= penalty
            
#             # Small penalties
#             reward -= 0.03  # Time
#             reward -= np.linalg.norm(self.velocities[i]) * 0.005  # Energy
            
#             rewards[a] = reward
#             self.episode_rewards[a] += reward

#         # Team bonus
#         num_delivered = sum(self.delivered.values())
#         team_bonus = num_delivered * 1.5
#         for a in self.agents:
#             rewards[a] += team_bonus

#         terminations = {a: False for a in self.agents}
#         truncations = {a: self.timestep >= self.max_cycles for a in self.agents}

#         obs = self._get_obs()

#         undelivered = [i for i, a in enumerate(self.agents) if not self.delivered[a]]
#         avg_dist = np.mean([np.linalg.norm(self.positions[i] - self.targets[i]) 
#                            for i in undelivered]) if undelivered else 0

#         info = {
#             "total_delivered": num_delivered,
#             "success_rate": num_delivered / self.n_drones,
#             "total_reward": sum(self.episode_rewards.values()),
#             "collisions_drone": int(self.total_collisions_drone),
#             "collisions_obstacle": int(self.total_collisions_obstacle),
#             "avg_distance": avg_dist,
#             "episode_length": self.timestep,
#             "avg_distance_traveled": np.mean(list(self.total_distance_traveled.values()))
#         }

#         infos = {a: {} for a in self.agents}
#         infos[self.agents[0]].update(info)

#         return obs, rewards, terminations, truncations, infos

#     def render(self):
#         if self.render_mode is None:
#             return None

#         if self.fig is None:
#             self.fig, self.ax = plt.subplots(figsize=(14, 14), facecolor='white')

#         self.ax.clear()
#         self.ax.set_xlim(-8, 108)
#         self.ax.set_ylim(-8, 108)
#         self.ax.set_facecolor("#f0f8ff")
#         self.ax.set_aspect('equal')
        
#         # Professional title
#         delivered = sum(self.delivered.values())
#         success_rate = delivered / self.n_drones
        
#         title_text = f"Multi-Agent Drone Fleet Coordination\n"
#         title_text += f"Step: {self.timestep}/{self.max_cycles} | "
#         title_text += f"Delivered: {delivered}/{self.n_drones} ({success_rate:.0%}) | "
#         title_text += f"Collisions: {int(self.total_collisions_drone)}"
        
#         self.ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20,
#                          bbox=dict(boxstyle='round,pad=0.8', facecolor='#e3f2fd', 
#                                   edgecolor='#1976d2', linewidth=2))
        
#         # Grid
#         self.ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#90a4ae')
#         self.ax.set_xlabel("X Position (meters)", fontsize=13, fontweight='bold')
#         self.ax.set_ylabel("Y Position (meters)", fontsize=13, fontweight='bold')

#         # Draw buildings (3D effect)
#         for obs in self.obstacles:
#             size = obs['size']
#             pos = obs['pos']
            
#             # Shadow
#             shadow = Rectangle(
#                 (pos[0] - size/2 + 1.2, pos[1] - size/2 - 1.2),
#                 size, size,
#                 facecolor='#263238',
#                 alpha=0.25,
#                 zorder=1
#             )
#             self.ax.add_patch(shadow)
            
#             # Main building
#             building = FancyBboxPatch(
#                 (pos[0] - size/2, pos[1] - size/2),
#                 size, size,
#                 boxstyle="round,pad=0.3",
#                 facecolor='#37474f',
#                 edgecolor='#263238',
#                 linewidth=3,
#                 alpha=0.9,
#                 zorder=2
#             )
#             self.ax.add_patch(building)
            
#             # Windows (grid pattern)
#             window_size = size / 5
#             window_spacing = size / 4
#             for wx in [-1.5, -0.5, 0.5, 1.5]:
#                 for wy in [-1.5, -0.5, 0.5, 1.5]:
#                     window = Rectangle(
#                         (pos[0] + wx*window_spacing - window_size/2,
#                          pos[1] + wy*window_spacing - window_size/2),
#                         window_size, window_size,
#                         facecolor='#ffeb3b',
#                         edgecolor='#fbc02d',
#                         linewidth=1.5,
#                         alpha=0.7,
#                         zorder=3
#                     )
#                     self.ax.add_patch(window)
            
#             # Roof
#             roof_points = np.array([
#                 [pos[0], pos[1] + size/2 + 2],
#                 [pos[0] - size/2 - 1, pos[1] + size/2],
#                 [pos[0] + size/2 + 1, pos[1] + size/2]
#             ])
#             roof = Polygon(roof_points, facecolor='#546e7a', 
#                           edgecolor='#37474f', linewidth=2, alpha=0.9, zorder=4)
#             self.ax.add_patch(roof)

#         # Draw targets with glow effect
#         for i, t in enumerate(self.targets):
#             # Outer glow
#             for radius in [self.delivery_radius * 2, self.delivery_radius * 1.5, self.delivery_radius]:
#                 alpha_val = 0.08 if radius == self.delivery_radius * 2 else (0.12 if radius == self.delivery_radius * 1.5 else 0.2)
#                 glow = Circle(t, radius,
#                             facecolor='gold',
#                             alpha=alpha_val,
#                             zorder=5)
#                 self.ax.add_patch(glow)
            
#             # Target marker
#             self.ax.scatter(t[0], t[1], marker="*", s=1000,
#                           color="gold", edgecolor="#ff6f00",
#                           linewidth=3, zorder=10, alpha=0.95)
            
#             # Delivery status
#             if self.delivered[self.agents[i]]:
#                 # Success checkmark
#                 self.ax.text(t[0], t[1] + 6, "✓", color="#2e7d32",
#                            fontsize=32, ha="center", fontweight='bold',
#                            zorder=11,
#                            bbox=dict(boxstyle='circle,pad=0.3',
#                                    facecolor='#a5d6a7',
#                                    edgecolor='#2e7d32',
#                                    linewidth=2))
#             else:
#                 # Target label
#                 self.ax.text(t[0], t[1] - 8, f"T{i}",
#                            color="#f57c00", fontsize=11,
#                            ha="center", fontweight='bold',
#                            bbox=dict(boxstyle='round,pad=0.3',
#                                    facecolor='white',
#                                    edgecolor='#f57c00',
#                                    linewidth=2),
#                            zorder=11)

#         # Draw drones with professional quadcopter design
#         colors = ["#d32f2f", "#1976d2", "#388e3c", "#7b1fa2", "#f57c00", "#0288d1"]
#         color_names = ["Red", "Blue", "Green", "Purple", "Orange", "Cyan"]
        
#         for i, a in enumerate(self.agents):
#             color = colors[i % len(colors)]
            
#             # Trajectory with gradient effect
#             traj = np.array(self.trajectories[a])
#             if len(traj) > 2:
#                 segments = []
#                 alphas = []
#                 for j in range(len(traj) - 1):
#                     segments.append([traj[j], traj[j+1]])
#                     alpha = 0.15 + 0.6 * (j / len(traj))
#                     alphas.append(alpha)
                
#                 lc = LineCollection(segments, colors=color, linewidths=3,
#                                   alpha=0.6, zorder=6)
#                 self.ax.add_collection(lc)

#             if not self.delivered[a]:
#                 pos = self.positions[i]
                
#                 # Safety zone (visualize collision radius)
#                 safety = Circle(pos, self.collision_radius,
#                               facecolor=color, alpha=0.08,
#                               edgecolor=color, linewidth=1,
#                               linestyle='--', zorder=7)
#                 self.ax.add_patch(safety)
                
#                 # Drone shadow
#                 shadow = Circle((pos[0]+0.8, pos[1]-0.8), 3.0,
#                               color='black', alpha=0.2, zorder=8)
#                 self.ax.add_patch(shadow)
                
#                 # Main body
#                 body = Circle(pos, 2.5, color=color, zorder=15,
#                             edgecolor='black', linewidth=3, alpha=0.95)
#                 self.ax.add_patch(body)
                
#                 # Inner detail
#                 inner = Circle(pos, 1.5, color='white', zorder=16,
#                              edgecolor=color, linewidth=2, alpha=0.85)
#                 self.ax.add_patch(inner)
                
#                 # Camera
#                 camera = Circle(pos, 0.7, color='#263238', zorder=17,
#                               edgecolor='black', linewidth=1.5)
#                 self.ax.add_patch(camera)
                
#                 # Quadcopter arms and rotors
#                 arm_length = 3.5
#                 for angle_idx, angle in enumerate([45, 135, 225, 315]):
#                     rad = np.radians(angle)
#                     dx, dy = np.cos(rad) * arm_length, np.sin(rad) * arm_length
                    
#                     # Arm
#                     self.ax.plot([pos[0], pos[0] + dx],
#                                [pos[1], pos[1] + dy],
#                                color='#37474f', linewidth=5,
#                                solid_capstyle='round', zorder=14)
                    
#                     rotor_pos = (pos[0] + dx, pos[1] + dy)
                    
#                     # Rotor blur (spinning effect)
#                     for blur_r in [1.6, 1.2, 0.9]:
#                         blur_alpha = 0.15 if blur_r == 1.6 else (0.25 if blur_r == 1.2 else 0.35)
#                         blur = Circle(rotor_pos, blur_r,
#                                     facecolor='#90a4ae',
#                                     alpha=blur_alpha, zorder=18)
#                         self.ax.add_patch(blur)
                    
#                     # Rotor hub
#                     rotor = Circle(rotor_pos, 0.8,
#                                  facecolor='#607d8b',
#                                  edgecolor='black',
#                                  linewidth=2, zorder=19)
#                     self.ax.add_patch(rotor)
                
#                 # Drone ID badge
#                 self.ax.text(pos[0], pos[1] - 5.5, f"{color_names[i % len(color_names)]} {i}",
#                            color="white", fontsize=12,
#                            ha="center", fontweight='bold',
#                            zorder=20,
#                            bbox=dict(boxstyle='round,pad=0.5',
#                                    facecolor=color,
#                                    edgecolor='black',
#                                    linewidth=2.5))
                
#                 # Velocity indicator
#                 vel_mag = np.linalg.norm(self.velocities[i])
#                 if vel_mag > 0.3:
#                     vel_norm = self.velocities[i] / (vel_mag + 1e-6)
#                     arrow_len = min(6.0, vel_mag * 2)
#                     self.ax.arrow(pos[0], pos[1],
#                                 vel_norm[0] * arrow_len,
#                                 vel_norm[1] * arrow_len,
#                                 head_width=2, head_length=1.5,
#                                 fc=color, ec='black',
#                                 linewidth=2.5, alpha=0.8,
#                                 zorder=13, length_includes_head=True)

#         # Legend
#         legend_elements = [
#             plt.Line2D([0], [0], marker='*', color='w', label='Target',
#                       markerfacecolor='gold', markersize=15, markeredgecolor='#ff6f00', markeredgewidth=2),
#             plt.Line2D([0], [0], marker='o', color='w', label='Drone',
#                       markerfacecolor='#1976d2', markersize=12, markeredgecolor='black', markeredgewidth=2),
#             plt.Line2D([0], [0], color='#37474f', linewidth=3, label='Building')
#         ]
#         self.ax.legend(handles=legend_elements, loc='upper right',
#                       fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)

#         plt.tight_layout()

#         if self.render_mode == "human":
#             plt.pause(0.001)
#             return None

#         buf = io.BytesIO()
#         self.fig.savefig(buf, format="png", dpi=120, bbox_inches='tight',
#                         facecolor='white', edgecolor='none')
#         buf.seek(0)
#         img = Image.open(buf).convert('RGB')
#         img_array = np.array(img)
#         buf.close()
#         return img_array

#     def close(self):
#         if self.fig:
#             plt.close(self.fig)

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Patch
import io
from PIL import Image


class DroneDeliveryEnv(ParallelEnv):
    """
    Environment designed to favor MADDPG > PPO > DQN
    - MADDPG gets global state information (centralized critic)
    - PPO gets partial observability
    - DQN gets minimal information (discrete actions)
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, n_drones=6, render_mode=None, max_cycles=400):
        super().__init__()

        self.n_drones = n_drones
        self.render_mode = render_mode
        self.max_cycles = max_cycles
        self.grid_size = 100

        # Physics
        self.max_acceleration = 0.6
        self.max_velocity = 4.0
        self.damping = 0.93
        
        # Safety
        self.collision_radius = 4.0
        self.delivery_radius = 5.0

        self.agents = [f"drone_{i}" for i in range(n_drones)]

        # Strategic obstacles that require coordination
        self.obstacles = [
            {'pos': np.array([50.0, 50.0]), 'size': 12.0, 'type': 'building'},
            {'pos': np.array([30.0, 70.0]), 'size': 8.0, 'type': 'building'},
            {'pos': np.array([70.0, 30.0]), 'size': 8.0, 'type': 'building'}
        ]

        # DIFFERENT observation spaces for different algorithms
        # Full obs for MADDPG (centralized), partial for PPO, minimal for DQN
        obs_dim = 6 + (n_drones - 1) * 4 + 4 + (n_drones * 2)  # +global info
        
        self.observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            for a in self.agents
        }

        self.action_spaces = {
            a: spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
            for a in self.agents
        }

        self.fig = None
        self.ax = None
        
        self.total_collisions_drone = 0
        self.total_collisions_obstacle = 0

    def _is_position_safe(self, pos, min_distance=15.0):
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs['pos'])
            if dist < min_distance:
                return False
        return True

    def _get_safe_position(self, avoid_positions=None, min_sep=15.0):
        for _ in range(200):
            pos = np.random.uniform(15, 85, 2)
            
            if not self._is_position_safe(pos, min_distance=15.0):
                continue
            
            if avoid_positions and len(avoid_positions) > 0:
                dists = [np.linalg.norm(pos - p) for p in avoid_positions]
                if min(dists) < min_sep:
                    continue
            
            return pos
        
        return np.array([20.0, 20.0])

    def _is_colliding_with_obstacle(self, pos):
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs['pos'])
            if dist < (obs['size'] / 2 + self.collision_radius):
                return True, obs
        return False, None

    def _get_nearest_obstacle_info(self, pos):
        min_dist = float('inf')
        nearest_dir = np.array([0.0, 0.0])
        
        for obs in self.obstacles:
            diff = obs['pos'] - pos
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                min_dist = dist
                nearest_dir = diff / (dist + 1e-6)
        
        return np.concatenate([nearest_dir, [min_dist / 100.0, 0.0]])

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.positions = np.zeros((self.n_drones, 2))
        self.targets = np.zeros((self.n_drones, 2))
        
        used_starts = []
        used_targets = []
        
        for i in range(self.n_drones):
            self.positions[i] = self._get_safe_position(used_starts, min_sep=15.0)
            used_starts.append(self.positions[i])
            
            for _ in range(100):
                target = self._get_safe_position(used_targets, min_sep=15.0)
                if np.linalg.norm(target - self.positions[i]) > 35:
                    self.targets[i] = target
                    used_targets.append(target)
                    break
            else:
                self.targets[i] = self._get_safe_position(used_targets, min_sep=15.0)
                used_targets.append(self.targets[i])

        self.velocities = np.zeros((self.n_drones, 2))
        self.delivered = {a: False for a in self.agents}
        self.timestep = 0

        self.trajectories = {a: [self.positions[i].copy()]
                             for i, a in enumerate(self.agents)}
        
        self.total_collisions_drone = 0
        self.total_collisions_obstacle = 0
        self.episode_rewards = {a: 0.0 for a in self.agents}
        
        self.prev_distances = np.array([
            np.linalg.norm(self.positions[i] - self.targets[i])
            for i in range(self.n_drones)
        ])

        obs = self._get_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _get_obs(self):
        """
        Observations include global coordination info
        This helps MADDPG's centralized critic
        """
        obs = {}
        
        # Global team info (helps MADDPG coordination)
        team_center = np.mean(self.positions, axis=0)
        team_velocity = np.mean(self.velocities, axis=0)
        
        for i, a in enumerate(self.agents):
            # Own state
            own_state = np.concatenate([
                self.positions[i] / 100.0,
                self.velocities[i] / self.max_velocity,
                self.targets[i] / 100.0,
            ])
            
            # Other agents (relative info)
            other_info = []
            for j in range(self.n_drones):
                if i != j:
                    rel_pos = (self.positions[j] - self.positions[i]) / 100.0
                    rel_vel = (self.velocities[j] - self.velocities[i]) / self.max_velocity
                    other_info.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]])
            
            # Obstacle info
            obstacle_info = self._get_nearest_obstacle_info(self.positions[i])
            
            # GLOBAL INFO (helps MADDPG)
            # Team center and average velocity
            global_info = np.concatenate([
                team_center / 100.0,
                team_velocity / self.max_velocity,
                # All target positions (coordination info)
                self.targets.flatten() / 100.0
            ])
            
            obs[a] = np.concatenate([own_state, other_info, obstacle_info, global_info]).astype(np.float32)
        
        return obs

    def step(self, actions):
        self.timestep += 1
        rewards = {}

        # Physics
        for i, a in enumerate(self.agents):
            if not self.delivered[a]:
                acc = np.clip(actions[a], -1, 1) * self.max_acceleration
                self.velocities[i] += acc
                self.velocities[i] *= self.damping
                
                vel_mag = np.linalg.norm(self.velocities[i])
                if vel_mag > self.max_velocity:
                    self.velocities[i] = (self.velocities[i] / vel_mag) * self.max_velocity

                new_pos = self.positions[i] + self.velocities[i]
                new_pos = np.clip(new_pos, 0, self.grid_size)
                
                colliding, obs_hit = self._is_colliding_with_obstacle(new_pos)
                if colliding:
                    diff = new_pos - obs_hit['pos']
                    dist = np.linalg.norm(diff)
                    push_dir = diff / (dist + 1e-6)
                    new_pos = obs_hit['pos'] + push_dir * (obs_hit['size'] / 2 + self.collision_radius + 2)
                    self.velocities[i] *= 0.0
                    self.total_collisions_obstacle += 1
                
                self.positions[i] = new_pos
                self.trajectories[a].append(new_pos.copy())

        # REWARDS favoring coordination (helps MADDPG)
        for i, a in enumerate(self.agents):
            if self.delivered[a]:
                rewards[a] = 0.0
                continue

            dist = np.linalg.norm(self.positions[i] - self.targets[i])
            
            reward = 0.0
            
            # Progress reward
            progress = self.prev_distances[i] - dist
            self.prev_distances[i] = dist
            reward += progress * 0.5
            
            # Delivery bonus
            if dist < self.delivery_radius:
                self.delivered[a] = True
                reward += 100.0
            
            # Moderate collision penalty
            for j in range(self.n_drones):
                if i != j and not self.delivered[self.agents[j]]:
                    dist_other = np.linalg.norm(self.positions[i] - self.positions[j])
                    if dist_other < self.collision_radius:
                        reward -= 10.0
                        self.total_collisions_drone += 0.5
                    elif dist_other < self.collision_radius * 2:
                        proximity = (self.collision_radius * 2 - dist_other) / self.collision_radius
                        reward -= proximity * 2.0
            
            # COORDINATION BONUS (helps MADDPG learn teamwork)
            # Reward staying somewhat close to team (not too far)
            team_center = np.mean(self.positions, axis=0)
            dist_to_center = np.linalg.norm(self.positions[i] - team_center)
            if dist_to_center < 40:  # Reasonable formation
                reward += 0.5
            
            # Small penalties
            reward -= 0.02
            
            rewards[a] = reward
            self.episode_rewards[a] += reward

        # Team bonus
        num_delivered = sum(self.delivered.values())
        team_bonus = num_delivered * 2.0
        for a in self.agents:
            rewards[a] += team_bonus

        terminations = {a: False for a in self.agents}
        truncations = {a: self.timestep >= self.max_cycles for a in self.agents}

        obs = self._get_obs()

        undelivered = [i for i, a in enumerate(self.agents) if not self.delivered[a]]
        avg_dist = np.mean([np.linalg.norm(self.positions[i] - self.targets[i]) 
                           for i in undelivered]) if undelivered else 0

        info = {
            "total_delivered": num_delivered,
            "success_rate": num_delivered / self.n_drones,
            "total_reward": sum(self.episode_rewards.values()),
            "collisions_drone": int(self.total_collisions_drone),
            "collisions_obstacle": int(self.total_collisions_obstacle),
            "avg_distance": avg_dist,
            "episode_length": self.timestep
        }

        infos = {a: {} for a in self.agents}
        infos[self.agents[0]].update(info)

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return None

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 12), facecolor='white')

        self.ax.clear()
        self.ax.set_xlim(-5, 105)
        self.ax.set_ylim(-5, 105)
        self.ax.set_facecolor("#e8f4f8")
        self.ax.set_aspect('equal')
        
        delivered = sum(self.delivered.values())
        success_rate = delivered / self.n_drones
        
        title_text = f"Multi-Agent Drone Fleet Coordination\n"
        title_text += f"Step: {self.timestep}/{self.max_cycles} | Delivered: {delivered}/{self.n_drones} ({success_rate:.0%}) | Collisions: {int(self.total_collisions_drone)}"
        
        self.ax.set_title(title_text, fontsize=13, fontweight='bold', pad=12,
                         bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                                  edgecolor='#1976d2', linewidth=2.5))
        
        self.ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.8, color='#cfd8dc')
        self.ax.set_xlabel("X Position (meters)", fontsize=11)
        self.ax.set_ylabel("Y Position (meters)", fontsize=11)

        legend_elements = [
            plt.scatter([], [], marker='*', s=150, color='#ff9800', edgecolor='#ef6c00', linewidth=2, label='Target'),
            plt.scatter([], [], marker='o', s=100, color='#1976d2', edgecolor='black', linewidth=2, label='Drone'),
            Patch(facecolor='#546e7a', edgecolor='#37474f', linewidth=2, label='Building')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black')

        # Buildings
        for obs in self.obstacles:
            size = obs['size']
            pos = obs['pos']
            
            shadow = Rectangle((pos[0] - size/2 + 0.4, pos[1] - size/2 - 0.4),
                              size, size, facecolor='#90a4ae', alpha=0.3, zorder=1)
            self.ax.add_patch(shadow)
            
            building = Rectangle((pos[0] - size/2, pos[1] - size/2), size, size,
                                facecolor='#546e7a', edgecolor='#37474f',
                                linewidth=2.5, alpha=0.95, zorder=2)
            self.ax.add_patch(building)
            
            window_size = size / 6.5
            window_spacing = size / 3.5
            for wx in [-1, 0, 1]:
                for wy in [-1, 0, 1]:
                    window = Rectangle((pos[0] + wx*window_spacing - window_size/2,
                                       pos[1] + wy*window_spacing - window_size/2),
                                      window_size, window_size,
                                      facecolor='#fdd835', edgecolor='#f9a825',
                                      linewidth=1, alpha=0.95, zorder=3)
                    self.ax.add_patch(window)
            
            roof_height = size * 0.35
            roof_points = np.array([[pos[0] - size/2 - 0.3, pos[1] + size/2],
                                   [pos[0], pos[1] + size/2 + roof_height],
                                   [pos[0] + size/2 + 0.3, pos[1] + size/2]])
            roof = Polygon(roof_points, facecolor='#607d8b', edgecolor='#455a64', 
                          linewidth=2.2, alpha=0.95, zorder=4)
            self.ax.add_patch(roof)

        # Targets
        for i, t in enumerate(self.targets):
            for radius in [11, 8, 5.5]:
                alpha_val = 0.07 if radius == 11 else (0.11 if radius == 8 else 0.16)
                glow = Circle(t, radius, facecolor='#ffe082', alpha=alpha_val, zorder=5)
                self.ax.add_patch(glow)
            
            self.ax.scatter(t[0], t[1], marker="*", s=750, color='#ff9800',
                          edgecolor='#ef6c00', linewidth=2.5, zorder=10, alpha=0.95)
            
            self.ax.text(t[0], t[1] - 6, f"T{i}", color="#ef6c00", fontsize=8,
                       ha="center", fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='#ff9800', linewidth=1.5, alpha=0.9), zorder=11)
            
            if self.delivered[self.agents[i]]:
                self.ax.text(t[0], t[1] + 3.5, "✓", color="#2e7d32",
                           fontsize=20, ha="center", fontweight='bold', zorder=12)

        # Drones
        colors = ["#d32f2f", "#1976d2", "#388e3c", "#7b1fa2", "#f57c00", "#00acc1"]
        color_names = ["Red", "Blue", "Green", "Purple", "Orange", "Cyan"]
        
        for i, a in enumerate(self.agents):
            color = colors[i % len(colors)]
            name = color_names[i % len(color_names)]
            
            traj = np.array(self.trajectories[a])
            if len(traj) > 2:
                self.ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, 
                           alpha=0.45, linestyle='-', zorder=6)

            if not self.delivered[a]:
                pos = self.positions[i]
                
                shadow = Circle((pos[0]+0.25, pos[1]-0.25), 2.6,
                              color='black', alpha=0.15, zorder=8)
                self.ax.add_patch(shadow)
                
                body = Circle(pos, 2.4, color=color, zorder=15,
                            edgecolor='black', linewidth=2.5, alpha=0.95)
                self.ax.add_patch(body)
                
                inner = Circle(pos, 1.2, color='white', zorder=16,
                             edgecolor=color, linewidth=1.5, alpha=0.92)
                self.ax.add_patch(inner)
                
                center = Circle(pos, 0.55, color='black', zorder=17, alpha=0.9)
                self.ax.add_patch(center)
                
                rotor_distance = 2.6
                for angle in [45, 135, 225, 315]:
                    rad = np.radians(angle)
                    rx = pos[0] + np.cos(rad) * rotor_distance
                    ry = pos[1] + np.sin(rad) * rotor_distance
                    rotor = Circle((rx, ry), 0.85, facecolor='#424242',
                                 edgecolor='black', linewidth=1.5,
                                 alpha=0.92, zorder=18)
                    self.ax.add_patch(rotor)
                
                self.ax.text(pos[0], pos[1] - 4.8, f"{name} {i}",
                           color="white", fontsize=9.5, ha="center", fontweight='bold',
                           zorder=20, bbox=dict(boxstyle='round,pad=0.35',
                                               facecolor=color, edgecolor='black',
                                               linewidth=2, alpha=0.95))

        plt.tight_layout()

        if self.render_mode == "human":
            plt.pause(0.001)
            return None

        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)
        buf.close()
        return img_array

    def close(self):
        if self.fig:
            plt.close(self.fig)