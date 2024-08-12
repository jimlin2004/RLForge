import torch as th
import random
import numpy as np

def detectDevice():
    return th.device("cuda:0" if th.cuda.is_available() else "cpu")

def setRandomSeed(seed: int = 42):
    '''
    幫助復現實驗成果
    將設定pytorch、numpy、python random的seed
    注意: 這不會設定gym的seed，所以environment的實驗可能會不同
    '''
    random.seed(seed) 
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed) 
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True