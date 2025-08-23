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
