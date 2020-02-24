import numpy as np

from radar import MonostaticRadar
from target import JMLSTarget


T = np.array([
    [0.98, 0.01, 0.01],
    [0.29, 0.7, 0.01],
    [0.29, 0.01, 0.7]
])
S = np.array([
    [0, 0],
    [1, 0],
    [0, 1]
])

x0 = np.array([0, 0, 1, 1])

r = MonostaticRadar()
t = JMLSTarget(x0=x0, T=T, S=S)





