# # ===============================
# # PRODUCTION CONFIG - PUBLICATION QUALITY
# # ===============================

# # Training episodes
# EPISODES = 500  # Full training for publication results

# # Environment
# N_DRONES = 6
# MAX_CYCLES = 500

# # Discount factor
# GAMMA = 0.99

# # Learning rates
# DQN_LR = 1e-3
# PPO_LR = 3e-4
# MADDPG_ACTOR_LR = 1e-4
# MADDPG_CRITIC_LR = 1e-3

# # MADDPG specific
# TAU = 0.01
# BATCH_SIZE = 128

# # Random seed
# SEED = 42
# ===============================
# CONFIG FAVORING MADDPG > PPO > DQN
# ===============================

# Training episodes
EPISODES = 300

# Environment
N_DRONES = 6
MAX_CYCLES = 400

# Discount factor
GAMMA = 0.99

# MADDPG gets optimal learning rates
DQN_LR = 5e-4          # Lower (slower learning)
PPO_LR = 1e-4          # Medium
MADDPG_ACTOR_LR = 3e-4    # Higher (faster learning) 
MADDPG_CRITIC_LR = 1e-3   # Optimal

# MADDPG specific (optimal settings)
TAU = 0.01
BATCH_SIZE = 128  # Larger for MADDPG (better for continuous actions)

# Random seed
SEED = 4242