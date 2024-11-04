import gymnasium as gym
from agent import Agent
import matplotlib.pyplot as plt

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("LunarLander-v3")


agent = Agent(
    state_dim=env.observation_space.shape[0],
    hidden_dim=256,
    action_dim=env.action_space.n,
    device='cpu',
)

# pretrain
# agent.load()

reward_per_step = []
reward_per_episode = []
entropy_step = []
value_step = []

for episode_i in range(500):
    state, info = env.reset()
    episode_return = 0
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
    
        if terminated or truncated:
            done = True
            
        agent.buffer.states.append(state)
        agent.buffer.actions.append(action)
        agent.buffer.rewards.append(reward)
        agent.buffer.next_states.append(next_state)
        agent.buffer.dones.append(done)
        
        state = next_state
        
        episode_return += reward
        reward_per_step.append(reward)
        entropy_step.append(agent.entropy)
        value_step.append(agent.value)
        
    print(f'{episode_i=} {episode_return=}')
    reward_per_episode.append(episode_return)
    
    agent.update()

    if episode_i % 100 == 0:
        agent.save()


agent.save()

plt.subplot(2, 2, 1)
plt.plot(reward_per_episode)
plt.title("reward_per_episode")

plt.subplot(2, 2, 2)
plt.plot(reward_per_step)
plt.title("reward_per_step")

plt.subplot(2, 2, 3)
plt.plot(entropy_step)
plt.title("entropy")

plt.subplot(2, 2, 4)
plt.plot(value_step)
plt.title("value")


plt.show()

env.close()

