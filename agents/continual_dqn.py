import torch
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer

class EWC:
    def __init__(self, model):
        self.model = model
        self.fisher = {}
        self.params_snapshot = {}

    def snapshot(self):
        self.params_snapshot = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def estimate_fisher(self, buffer: ReplayBuffer, batch_size=256, iters=50, device="cpu"):
        fisher = {n: torch.zeros_like(p, device=device) for n, p in self.model.named_parameters() if p.requires_grad}
        mse = torch.nn.MSELoss()
        for _ in range(iters):
            if len(buffer) < batch_size:
                break
            s, a, r, s2, d = buffer.sample(batch_size)
            s = torch.tensor(s, dtype=torch.float32, device=device)
            a = torch.tensor(a, dtype=torch.long, device=device)
            self.model.zero_grad(set_to_none=True)
            q = self.model(s).gather(1, a.view(-1,1)).squeeze(1)
            loss = (q**2).mean()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
        for n in fisher:
            fisher[n] /= max(1, iters)
        self.fisher = fisher

    def penalty(self, model, lambda_ewc=1000.0):
        if not self.fisher or not self.params_snapshot:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params_snapshot[n]).pow(2)).sum()
        return lambda_ewc * loss
