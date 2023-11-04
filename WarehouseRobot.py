import gym
import numpy as np
from gym import spaces

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=5, items_count=3):
        super(WarehouseEnv, self).__init__()
        # Define action and observation space
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space is the flattened grid plus the robot's position
        self.observation_space = spaces.Box(
            low=0, high=1, shape=((grid_size ** 2) + 2,), dtype=np.float32
        )

        self.grid_size = grid_size
        self.items_count = items_count
        # Initialize the state
        self.state = None
        self.items_collected = 0
        self.drop_off_point = (grid_size - 1, grid_size - 1)  # bottom-right corner
        self.robot_position = None
        self.items_positions = []

    def reset(self):
        # Reset the environment state
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.items_collected = 0
        # Place items at random positions
        self.items_positions = [
            tuple(np.random.choice(range(self.grid_size), size=2)) for _ in range(self.items_count)
        ]
        for item in self.items_positions:
            self.state[item] = 1  # 1 represents an item

        # Place drop-off point
        self.state[self.drop_off_point] = 2  # 2 represents the drop-off point

        # Randomly place the robot in an empty space
        empty_spaces = np.argwhere(self.state == 0)
        self.robot_position = tuple(empty_spaces[np.random.choice(len(empty_spaces))])
        return self._get_observation()

    def _get_observation(self):
        # Return the flattened grid and the robot's position as the observation
        return np.concatenate((self.state.flatten(), np.array(self.robot_position)))

    def step(self, action):
        # Update the robot's position based on the action
        if action == 0 and self.robot_position[0] > 0:
            self.robot_position = (self.robot_position[0] - 1, self.robot_position[1])
        elif action == 1 and self.robot_position[0] < self.grid_size - 1:
            self.robot_position = (self.robot_position[0] + 1, self.robot_position[1])
        elif action == 2 and self.robot_position[1] > 0:
            self.robot_position = (self.robot_position[0], self.robot_position[1] - 1)
        elif action == 3 and self.robot_position[1] < self.grid_size - 1:
            self.robot_position = (self.robot_position[0], self.robot_position[1] + 1)

        # Check for item collection
        if self.robot_position in self.items_positions:
            self.items_collected += 1
            self.items_positions.remove(self.robot_position)
            self.state[self.robot_position] = 0  # Remove the item from the grid

        # Check if all items are collected and robot reached the drop-off point
        done = self.items_collected == self.items_count and self.robot_position == self.drop_off_point
        reward = 1 if done else -0.1  # Reward for completing the task, small penalty otherwise

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        # Simple rendering: print the grid and robot position
        grid_copy = self.state.copy()
        grid_copy[self.robot_position] = 3  # 3 represents the robot
        print("Grid:")
        print(grid_copy)
        print(f"Robot Position: {self.robot_position}, Items Collected: {self.items_collected}")

class QLearningAgent:
    def __init__(self, action_space, state_space, learning_rate=0.01, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        # Initialize Q-table, rows: states, columns: actions
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def learn(self, state, action, reward, next_state):
        # Update Q-table using the Q-learning algorithm
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (target - predict)
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Training loop
def train_agent(env, agent, episodes):
    rewards = []
    for episode in range(episodes):
        print(episode)
        state = env.reset()
        state = np.argmax(state[:-2])  # Get the index of the robot's position in the flattened grid
        total_rewards = 0

        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.argmax(next_state[:-2])
            agent.learn(state, action, reward, next_state)

            state = next_state
            total_rewards += reward
            env.render()

        rewards.append(total_rewards)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}: Average Reward: {avg_reward}")

    return rewards

env = WarehouseEnv(grid_size=5, items_count=3)
state_space = (env.grid_size ** 2) * env.items_count  
action_space = env.action_space.n
agent = QLearningAgent(action_space, state_space)

episodes = 1000
rewards = train_agent(env, agent, episodes)