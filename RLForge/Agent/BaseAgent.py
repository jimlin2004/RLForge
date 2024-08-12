import torch as th
import abc
from RLForge.Env.Env import Env

class BaseAgent(object):
    def __init__(self, env: Env, device: th.device, lr = 3e-4):
        super().__init__()
        self.n_state = env.n_state
        self.n_action = env.n_action
        self.lr = lr
        self.env = env
        self.device = device
    @abc.abstractmethod
    def runOneEpisode(self):
        raise NotImplementedError
    @abc.abstractmethod
    def learn(self, totalTimestep: int):
        raise NotImplementedError
    @abc.abstractmethod
    def train(self):
        '''
        Train agent,
        this method will be difference according to RL algorithm
        '''
        raise NotImplementedError
    @abc.abstractmethod
    def selectAction(self, state: th.Tensor):
        '''
        param:
            state: environment's state
        Agent select action according to the state it saw
        '''
        raise NotImplementedError
    @abc.abstractmethod
    def storeTransition(self, s, a, ns, r, d):
        raise NotImplementedError
    @abc.abstractmethod
    def save(self, dirPath: str):
        '''
        param:
            dirPath: 欲存模型的資料夾路徑，ex: "./saved"，最後面不需要'/'
        '''