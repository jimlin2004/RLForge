from RLForge.Agent.BaseAgent import BaseAgent
from .QNN import QNN
from .ReplayBuffer import ReplayBuffer
from RLForge import Logger
from RLForge import Env
import torch as th
import numpy as np

class DQN(BaseAgent):
    '''
    基礎的DQN
    '''
    def __init__(self, env: Env, device: th.device, n_hidden = 32, lr = 3e-4,
                gamma = 0.99, initEpsilon = 1, endEpsilon = 0.001,
                epsilonDecay = 1000, batchSize = 64, replayBufferSize = 10000,
                initReplayBufferSize = 200, optimizer = th.optim.Adam, maxEpisodeTimestep = None,
                trainInterval = 1, targetUpdateInterval = 200):
        '''
        param:
            n_hidden: 預設中QNN的linear layer神經元數
            lr: agent的learning rate, in [0, 1]
            gamma: bellman equation的discount factor, in [0, 1]
            initEpsilon: DQN中的初始探索參數, in [0, 1]
            endEpsilon: DQN中探索參數的最小值, in [0, 1]
            epsilonDecay: 探索參數epsilon在經過多少次timestep後線性下降到endEpsilon
            batchSize: 訓練時要從replay buffer中一batch抽多少資料出來
            replayBufferSize: replay buffer的size
            initReplayBufferSize: replay buffer要收集多少資料後開始訓練
            optimizer: 梯度更新時要用的optimizer
            maxEpisodeTimestep: 一局epsiode最多跑多少次
            trainInterval: DQN進行梯度更新的間隔
            targetUpdateInterval: DQN Q_target進行更新的間隔
        '''
        super().__init__(env, device, lr)
        self.Q = QNN(env.n_state, env.n_action, n_hidden).to(device)
        self.Q_target = QNN(env.n_state, env.n_action, n_hidden).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.gamma = gamma
        self.epsilon = initEpsilon
        self.epsiloneDelta = (initEpsilon - endEpsilon) / epsilonDecay
        self.endEpsilon = endEpsilon
        self.batchSize = batchSize
        self.replayBuffer = ReplayBuffer(env.n_state, replayBufferSize)
        self.initReplayBufferSize = initReplayBufferSize
        self.optimizer = optimizer(self.Q.parameters(), self.lr)
        self.mse = th.nn.MSELoss()
        self.trainInterval = trainInterval
        self.targetUpdateInterval = targetUpdateInterval
        self.logger = Logger()
        self.currTimestep = 0
        self.trainTimestep = 0

    def selectAction(self, state: th.Tensor):
        if (np.random.uniform(0, 1) > self.epsilon):
            self.Q.train(False)
            action_Q = self.Q(state)
            action = th.argmax(action_Q).item()
        else:
            action = np.random.randint(0, self.n_action)
        return action
    
    def storeTransition(self, s, a, ns, r, d):
        self.replayBuffer.push(s, a, ns, r, d)
        
    def runOneEpisode(self):
        state = self.env.reset()[0]
        ep_reward = 0
        ep_loss = 0
        ep_t = 1
        for ep_t in range(1, self.env.maxEpisodeTimestep + 1):
            stateTensor = th.tensor(state, dtype = th.float32).unsqueeze(0).to(self.device)
            action = self.selectAction(stateTensor)
            nextState, reward, done, _, _ = self.env.step(action)
            ep_reward += reward
            self.storeTransition(state, action, nextState, reward, done)
            self.epsilon = max(self.epsilon - self.epsiloneDelta, self.endEpsilon)
            if ((len(self.replayBuffer) >= self.initReplayBufferSize) and (self.currTimestep % self.trainInterval == 0)):
                self.trainTimestep += 1
                self.train()
            if (done):
                break
            state = nextState
        self.logger["Episode/ep_reward"] = ep_reward
        self.logger["Episode/ep_timestep"] = ep_t
        return ep_t
    
    def train(self):
        self.Q.train(True)
        self.Q.train(True)
        states, actions, nextStates, rewards, dones = self.replayBuffer.sample(self.batchSize)
        states = th.tensor(states, dtype = th.float32).to(self.device)
        actions = th.tensor(actions, dtype = th.int64).to(self.device)
        nextStates = th.tensor(nextStates, dtype = th.float32).to(self.device)
        rewards = th.tensor(rewards, dtype = th.float32).to(self.device)
        dones = th.tensor(dones, dtype = th.int32).to(self.device)
        
        Q_eval = self.Q(states).gather(1, actions)
        with th.no_grad():
            maxQ_next = self.Q_target(nextStates).max(1)[0]
            maxQ_next = maxQ_next.reshape(-1, 1)
            Q_target = rewards + (1 - dones) * self.gamma * maxQ_next
        loss = self.mse(Q_eval, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if (self.trainTimestep % self.targetUpdateInterval == 0):
            self.Q_target.load_state_dict(self.Q.state_dict())
        return loss.item()

    def learn(self, totalTimestep: int):
        episodeCnt = 0
        while (self.currTimestep < totalTimestep):
            episodeCnt += 1
            self.currTimestep += self.runOneEpisode()
            self.logger["time/episode"] = episodeCnt
            self.logger.summary()