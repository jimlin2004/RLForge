import RLForge

if (__name__ == "__main__"):
    device = RLForge.detectDevice()
    env = RLForge.Env(device, "CartPole-v1")
    agent = RLForge.DQN(env, device)
    agent.learn(10000)