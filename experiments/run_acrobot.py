from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.evaluation import plot_curves
import gymnasium as gym

def train_acrobot():
    env = gym.make("Acrobot-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    buffer = ReplayBuffer()
    rewards = []

    for ep in range(500):
        s, _ = env.reset()
        total = 0
        done = False
        while not done:
            a = agent.act(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buffer.push(s, a, r, ns, float(done))
            s = ns
            total += r
        rewards.append(total)
        agent.step_eps()
    env.close()
    plot_curves({"Acrobot": rewards}, "Acrobot Returns", savepath="results/acrobot.png")

if __name__ == "__main__":
    train_acrobot()
