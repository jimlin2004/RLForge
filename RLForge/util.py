import torch as th

def detectDevice():
        return th.device("cuda:0" if th.cuda.is_available() else "cpu")