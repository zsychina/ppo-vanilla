import gym
from agent import Agent
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2", render_mode="human")


agent = Agent(
    state_dim=env.observation_space.shape[0],
    hidden_dim=1024,
    action_dim=4,
    device='mps'
)

agent.load()

for episode_i in range(10):
    state, info = env.reset()
    episode_return = 0
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
    
        if terminated or truncated:
            done = True
        
        state = next_state
        
        episode_return += reward
        
    print(f'{episode_i=} {episode_return=}')

env.close()