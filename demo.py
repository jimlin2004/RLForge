import RLForge

if (__name__ == "__main__"):
    device = RLForge.detectDevice()
    env = RLForge.Env("CartPole-v1", device)
    agent = RLForge.DQN(env, device)
    agent.learn(10000)