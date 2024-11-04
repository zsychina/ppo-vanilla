import torch
import numpy as np
from mlp import MLP


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        

class Agent:
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 lmbda=0.90,
                 gamma=0.90,
                 eps_clip=0.2,
                 K_epochs=10,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 device='cpu',
                 model_path='./checkpoint/'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lmbda = lmbda
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr_actor = lr_actor  
        self.lr_critic = lr_critic
        self.buffer = RolloutBuffer()
        self.device = device
        self.model_path = model_path
        
        self.actor = MLP(state_dim=state_dim, hidden_dim=hidden_dim, out_dim=action_dim).to(device)
        self.critic = MLP(state_dim=state_dim, hidden_dim=hidden_dim, out_dim=1).to(device)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        

        self.mse = torch.nn.MSELoss()
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = state.unsqueeze(0)  # [1, state_dim]
        actor_out = self.actor(state)
        action_dist = torch.softmax(actor_out, dim=-1)
        action_dist = torch.distributions.Categorical(action_dist)
        action = action_dist.sample()
        
        # debug
        self.entropy = action_dist.entropy().item()
        self.value = self.critic(state).item()
        
        return action.item()
        
    def update(self):
        
        # solve extremely low
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float).reshape(-1, self.state_dim).to(self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.int64).reshape(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float).reshape(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(self.buffer.next_states), dtype=torch.float).reshape(-1, self.state_dim).to(self.device)
        dones = torch.tensor(np.array(self.buffer.dones), dtype=torch.float).reshape(-1, 1).to(self.device)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        
        advantage = 0
        advantage_list = []
        td_delta = td_delta.detach().cpu().numpy()
        for delta in td_delta[::-1]:
            advantage = delta + self.gamma * self.lmbda * advantage
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_list = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)
        
        old_log_probs = torch.log(torch.softmax(self.actor(states), dim=-1).gather(-1, actions)).detach()
        
        for _ in range(self.K_epochs):
            log_probs = torch.log(torch.softmax(self.actor(states), dim=-1).gather(-1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage_list
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage_list
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                self.mse(self.critic(states), td_target.detach())
            )
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

        self.buffer.clear()
        
    def save(self):
        torch.save(self.actor.state_dict(), self.model_path+'actor.pt')
        torch.save(self.critic.state_dict(), self.model_path+'critic.pt')
    
    def load(self):
        self.actor.load_state_dict(torch.load(self.model_path+'actor.pt', map_location=lambda storage, loc: storage))
        
        self.critic.load_state_dict(torch.load(self.model_path+'critic.pt', map_location=lambda storage, loc: storage))
    
