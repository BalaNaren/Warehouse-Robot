import gym
import numpy as np
from gym import spaces

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=5, items_count=3):
        super(WarehouseEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=((grid_size ** 2) + 2,), dtype=np.float32
        )

        self.grid_size = grid_size
        self.items_count = items_count
        self.state = None
        self.items_collected = 0
        self.drop_off_point = (grid_size - 1, grid_size - 1)
        self.robot_position = None
        self.items_positions = []

    def reset(self):
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.items_collected = 0
        self.items_positions = [
            tuple(np.random.choice(range(self.grid_size), size=2)) for _ in range(self.items_count)
        ]
        for item in self.items_positions:
            self.state[item] = 1
        self.state[self.drop_off_point] = 2 
        empty_spaces = np.argwhere(self.state == 0)
        self.robot_position = tuple(empty_spaces[np.random.choice(len(empty_spaces))])
        return self._get_observation()

    def _get_observation(self):
        return np.concatenate((self.state.flatten(), np.array(self.robot_position)))

    def step(self, action):
        if action == 0 and self.robot_position[0] > 0:
            self.robot_position = (self.robot_position[0] - 1, self.robot_position[1])
        elif action == 1 and self.robot_position[0] < self.grid_size - 1:
            self.robot_position = (self.robot_position[0] + 1, self.robot_position[1])
        elif action == 2 and self.robot_position[1] > 0:
            self.robot_position = (self.robot_position[0], self.robot_position[1] - 1)
        elif action == 3 and self.robot_position[1] < self.grid_size - 1:
            self.robot_position = (self.robot_position[0], self.robot_position[1] + 1)
        if self.robot_position in self.items_positions:
            self.items_collected += 1
            self.items_positions.remove(self.robot_position)
            self.state[self.robot_position] = 0 
        done = self.items_collected == self.items_count and self.robot_position == self.drop_off_point
        reward = 1 if done else -0.1 - 0.01 

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        grid_copy = self.state.copy()
        grid_copy[self.robot_position] = 3 
        print("Grid:")
        print(grid_copy)
        print(f"Robot Position: {self.robot_position}, Items Collected: {self.items_collected}")

class QLearningAgent:
    def __init__(self, action_space, state_space, learning_rate=0.01, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space) 
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (target - predict)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_agent(env, agent, episodes, max_steps_per_episode):
    rewards = []
    for episode in range(episodes):
        print("In Episode: "+str(episode))
        state = env.reset()
        state_idx = np.argmax(state[:-2])
        total_rewards = 0
        step = 0
        done = False
        while not done and step < max_steps_per_episode:
            action = agent.choose_action(state_idx)
            next_state, reward, done, _ = env.step(action)
            next_state_idx = np.argmax(next_state[:-2])
            agent.learn(state_idx, action, reward, next_state_idx)
            state_idx = next_state_idx
            total_rewards += reward
            step += 1
            env.render()
        rewards.append(total_rewards)
        print(f"Episode {episode + 1}: Total Reward: {total_rewards}")
    return rewards

env = WarehouseEnv(grid_size=5, items_count=3)
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
agent = QLearningAgent(action_space, state_space)
episodes = 1000
max_steps_per_episode = 100 
rewards = train_agent(env, agent, episodes, max_steps_per_episode)