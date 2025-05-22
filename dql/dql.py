import torch
import gymnasium as gym 
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple,deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f 
from IPython import display
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#%% env
env=gym.make("CartPole-v1",render_mode="human")
device="cpu"
transition=namedtuple("transition",
                      ("state","action","next_state","reward"))

# %% replay memory
class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory=deque([],maxlen=capacity)
        
        
        
    def push(self,*args):
        self.memory.append(transition(*args))
        
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
        
        
        
    def __len__(self):
        return len(self.memory)
    
# %% model oluşturma    
class DQN(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQN,self).__init__()
        
        self.layer1=nn.Linear(n_observations,128)
        self.layer2=nn.Linear(128,128)
        self.layer3=nn.Linear(128,n_actions)
    def forward(self,x):
        x=f.relu(self.layer1(x))
        x=f.relu(self.layer2(x))
        return self.layer3(x)
    
#hiperparametreler   
batch_size=128
gamma=0.99
eps_start=0.9
eps_end=0.05
eps_decay=1000
tau=0.005 #update rate of target network
lr=1e-4
n_actions=env.action_space.n 
state,info= env.reset()
n_observations=len(state)
policy_net=DQN(n_observations,n_actions).to(device)
target_net=DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer=optim.Adam(policy_net.parameters(),lr=lr)
memory=ReplayMemory(10000)
steps_done=0

# %% hareket seçme
def select_act(state):
    global steps_done
    sample=random.random()
    eps_thresh=eps_end+(eps_start-eps_end)*math.exp(-1*steps_done/eps_decay)
    steps_done +=1 
    if sample>eps_thresh:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]],device=device,dtype=torch.long)
    
    
       
# %% görsellleştirme

episode_durations=[]
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t=torch.tensor(episode_durations,dtype=torch.float)
    if show_result:
        plt.title("result")
    else:
        plt.clf()
        plt.title("training...")
        
    plt.xlabel("episode")
    plt.ylabel("duration")
    plt.plot(durations_t.numpy())
    
    if len(durations_t)>100:
        means=durations_t.unfold(0,100,1).mean(1).view(-1)
        means=torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    display.display(plt.clf())
    display.clear_output(wait=True)
def optimize_model():
    if len(memory)<batch_size:
        return
    transitions=memory.sample(batch_size)
    batch=transition(*zip(*transitions))
    non_final_mask=torch.tensor(tuple(map(lambda s:s is not None,batch.next_state)),
                                device=device,dtype=torch.bool)
    non_final_next=torch.cat([s for s in batch.next_state if s is not None])
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    reward_batch=torch.cat(batch.reward)
    
    
    state_action_val=policy_net(state_batch).gather(1,action_batch)   
    next_state_val=torch.zeros(batch_size,device=device)
    with torch.no_grad():
        next_state_val[non_final_mask]=target_net(non_final_next).max(1).values
        
    exp_state_action=(next_state_val*gamma)+reward_batch
    ctriterion=nn.SmoothL1Loss()
    loss=ctriterion(state_action_val,exp_state_action.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
# %% model eğitimi
num_epsiodes=250
for i_episode in range(num_epsiodes):
    state,info=env.reset()
    state=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
    for t in count():
        action=select_act(state)
        observation,reward,terminated,truncated,_=env.step(action.item())
        reward=torch.tensor([reward],device=device)
        done=terminated or truncated
        if terminated:
            next_state=None
        else:
            next_state=torch.tensor(observation,device=device,dtype=torch.float32).unsqueeze(0)
        memory.push(state,action,next_state,reward)
        state=next_state
         
        
        optimize_model()
        target_net_dict=target_net.state_dict()
        policy_net_dict=policy_net.state_dict()
        
        
        for key in policy_net_dict:
            target_net_dict[key]=policy_net_dict[key]*tau+target_net_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_dict)
            
        
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break   
print("done")
plot_durations(show_result=True)
plt.ioff()
plt.show()         
         
        
        
    



