# Continual Reinforcement Learning Experiments

This repository contains a collection of **toy experiments and implementations in Continual Reinforcement Learning (CRL)**.  
The goal is to explore methods that enable RL agents to learn across multiple tasks sequentially while mitigating **catastrophic forgetting** and improving **generalization**.

---

## 🎯 Motivation
Traditional RL assumes training on a single task/environment until convergence.  
However, real-world agents (e.g., robots) must **adapt continually to new tasks** without forgetting previously learned behaviors.  
This repo provides hands-on implementations of CRL methods and experiments to better understand:

- **Catastrophic forgetting** in RL  
- **Replay-based strategies**  
- **Regularization-based strategies (e.g., EWC, SI)**  
- **Latent representations for generalization across tasks**  

---

## 🧩 Implemented Experiments
- [x] Baseline: DQN/Policy Gradient on sequential CartPole → MountainCar  
- [ ] Experience Replay buffer across tasks  
- [ ] Elastic Weight Consolidation (EWC) for CRL  
- [ ] Latent-space representations for transfer across tasks  

---

## 📂 Repo Structure
```bash
continual-rl-experiments/
├── experiments/
│   ├── cartpole_mountaincar.py   # Baseline: sequential training (DQN)
│   └── ewc.py                    # Elastic Weight Consolidation (WIP)
│
├── docs/
│   └── example_plot.png          # Example learning curve
│
├── README.md                     # Project overview
├── requirements.txt              # Dependencies
└── .gitignore
```

---
## 🚀 Usage
```bash
# Clone the repo
git clone https://github.com/cajifan/continual-rl-experiments.git
cd continual-rl-experiments

# Install dependencies
pip install -r requirements.txt


# Run the baseline experiments
python experiments/cartpole_mountaincar.py --method baseline
```
