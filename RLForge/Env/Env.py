import gymnasium as gym
import torch as th

class Env(object):
    def __init__(self, device: th.device, gameName: str, render_mode = "rgb_array", maxEpisodeTimestep = None):
        if (maxEpisodeTimestep is None):
            self.env = gym.make(gameName, render_mode = render_mode)
        else:
            self.env = gym.make(gameName, render_mode = render_mode, max_episode_steps = maxEpisodeTimestep)
        self.maxEpisodeTimestep = self.env._max_episode_steps
        self.device = device
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    @property
    def n_action(self):
        return self.env.action_space.n
    @property
    def n_state(self):
        return self.env.observation_space.shape[0]