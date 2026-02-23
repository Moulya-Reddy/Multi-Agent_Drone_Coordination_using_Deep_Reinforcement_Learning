# 🚁 Multi-Agent Drone Coordination using Deep Reinforcement Learning

## 📌 Project Overview
This project presents a Multi-Agent Deep Reinforcement Learning (MADRL) framework for autonomous coordination of multiple drones in a shared environment. The system enables drones to learn efficient navigation, collision avoidance, and cooperative decision-making through continuous interaction with a simulated environment.

Each drone is modeled as an intelligent agent that observes its surroundings, selects optimal actions, and improves performance using reward-based learning. The system is designed to operate without centralized control, ensuring scalability and robustness in dynamic environments.

---

## 🎯 Objectives
- Develop an autonomous multi-drone coordination system  
- Implement reinforcement learning for intelligent navigation  
- Avoid collisions and optimize path efficiency  
- Compare performance of different RL algorithms  
- Improve energy efficiency and convergence speed  
- Demonstrate decentralized cooperative decision-making  

---

## 🧠 Algorithms Implemented
The following reinforcement learning algorithms are implemented and compared:

- **Deep Q-Network (DQN)** – Value-based learning for decision making  
- **Proximal Policy Optimization (PPO)** – Stable policy optimization  
- **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** – Cooperative multi-agent learning  

MADDPG is used as the primary algorithm due to its ability to support centralized training and decentralized execution, enabling efficient multi-drone coordination.

---

## 🏗️ System Architecture
The system follows a reinforcement learning workflow:

**Drone Agents → Environment → State Observation → Action Selection → Reward → Learning Update**

Each drone:
1. Observes its environment  
2. Selects an action using RL policy  
3. Moves within the environment  
4. Receives reward/penalty  
5. Updates its policy  
6. Improves performance over time  

---

## ⚙️ Tech Stack
- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- Pygame  
- Reinforcement Learning  
- Multi-Agent Systems  

---

## 📂 Project Structure
```

run_project.py        → Main execution file
quick_demo.py         → Quick simulation demo
custom_demo.py        → Custom simulation
config.py             → Hyperparameters
replay_buffer.py      → Experience replay memory

agents/               → DQN, PPO, MADDPG agents
environment/          → Drone simulation environment
training/             → Training scripts
simulations/          → Visualization scripts
results/              → Graphs and simulation outputs
models/               → Saved trained models

```

---

## ▶️ Installation & Setup

### 1. Clone repository
```

git clone [https://github.com/yourusername/Multi-Agent-Drone-RL](https://github.com/yourusername/Multi-Agent-Drone-RL)
cd Multi-Agent-Drone-RL

```

### 2. Install dependencies
```

pip install -r requirements.txt

```

---

## ▶️ How to Run

### Run full project
```

python run_project.py

```

### Run quick demo
```

python quick_demo.py

```

### Run custom simulation
```

python custom_demo.py

```

---

## 📊 Results
The system demonstrates effective multi-drone coordination and learning:

- Increased cumulative reward over training  
- Reduced collision rate  
- Efficient path planning  
- Stable convergence  
- Cooperative drone navigation  

Simulation GIFs and performance graphs are generated for:
- DQN  
- PPO  
- MADDPG  

MADDPG shows the best performance in multi-agent coordination.

---

## 📈 Evaluation Metrics
- Cumulative reward  
- Success rate  
- Collision rate  
- Energy consumption  
- Convergence speed  

---

## 🚀 Applications
- Autonomous delivery drones  
- Disaster response and rescue  
- Surveillance and monitoring  
- Smart city operations  
- Environmental monitoring  
- Traffic and crowd analysis  

---

## 🔬 Research Contribution
This project demonstrates how multi-agent reinforcement learning can be applied to drone fleet coordination in dynamic environments. The decentralized learning framework enables scalable and intelligent autonomous navigation suitable for real-world deployment.

