"""
QUICK VISUAL PREVIEW
See the realistic drone rendering without training
"""

import numpy as np
import matplotlib.pyplot as plt
from environment.drone_env import DroneDeliveryEnv

print("\n" + "="*70)
print("🎨 REALISTIC DRONE VISUALIZATION PREVIEW")
print("="*70 + "\n")

# Create environment
env = DroneDeliveryEnv(n_drones=6, render_mode="human", max_cycles=800)

# Reset with custom positions for good view
obs, _ = env.reset()

# Set some nice positions to showcase drones
env.positions = np.array([
    [20, 20],   # Bottom-left
    [80, 20],   # Bottom-right
    [20, 80],   # Top-left
    [80, 80],   # Top-right
    [35, 50],   # Mid-left
    [65, 50]    # Mid-right
])

env.targets = np.array([
    [80, 80],   # Opposite corners
    [20, 80],
    [80, 20],
    [20, 20],
    [65, 30],
    [35, 70]
])

# Set some velocities to show motion arrows
env.velocities = np.array([
    [1.5, 1.0],   # Moving up-right
    [-1.0, 1.5],  # Moving up-left
    [1.2, -0.8],  # Moving down-right
    [-1.5, -1.2], # Moving down-left
    [2.0, -1.5],  # Moving down-right fast
    [-0.8, 1.8]   # Moving up-left fast
])

# Add some trajectory history
for i, agent in enumerate(env.agents):
    # Create a path from start to current position
    start = env.positions[i] - env.velocities[i] * 10
    path = []
    for t in range(15):
        point = start + (env.positions[i] - start) * (t / 15)
        path.append(point)
    env.trajectories[agent] = path

print("Features you'll see:")
print("  ✓ 3 realistic building obstacles with windows")
print("  ✓ Quadcopter drones with 4 rotors")
print("  ✓ Spinning rotor blur effects")
print("  ✓ Landing gear")
print("  ✓ Camera/sensor pod")
print("  ✓ Velocity arrows showing direction")
print("  ✓ Speed indicators (m/s)")
print("  ✓ Gradient trajectory trails")
print("  ✓ 3D shadow effects")
print("  ✓ Gold star targets")
print("\nRendering...")

# Render the scene
env.render()

print("\n✅ Close the matplotlib window when done viewing\n")
print("This is how drones will look during training and demos!")
print("="*70 + "\n")

# Keep window open
plt.show()

env.close()