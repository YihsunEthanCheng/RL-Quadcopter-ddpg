%load_ext autoreload
%autoreload 2

from task import Task
from agents.agent_ddpg import DDPG
import numpy as np
#%%

# select a task: hover at [0,5,5] for 5 seconds 
init_pose = np.array([0., 0., 5., 0., 0., 0.])  # initial pose
task = Task(init_pose = init_pose, runtime = 5.)
agent = DDPG(task, '0918') 

#%%
