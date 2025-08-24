import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 device="cpu"):
        self.device = device
        self.q = QNetwork(state_dim, action_dim).to(device)
        self.target = QNetwork(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay

    def act(self, state):
        if random.random() < self.eps:
            return random.randrange(self.q.net[-1].out_features)
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q(s_t).argmax(dim=1).item()

    def update(self, batch, loss_fn=nn.MSELoss(), targets=None):
        s, a, r, s2, d = batch
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).view(-1,1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_values = self.q(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            y = r + (1 - d) * self.gamma * self.target(s2).max(dim=1)[0]

        loss = loss_fn(q_values, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def step_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_end)
