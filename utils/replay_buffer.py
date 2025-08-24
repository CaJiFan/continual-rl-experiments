import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def extend(self, tuples):
        self.buf.extend(tuples)

    def sample(self, batch=64):
        batch = random.sample(self.buf, batch)
        s, a, r, s2, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d)

    def __len__(self):
        return len(self.buf)
