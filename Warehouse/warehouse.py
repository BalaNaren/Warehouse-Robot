import logging
from collections import defaultdict, OrderedDict
import gym
from gym import spaces
from enum import Enum
import numpy as np
from typing import List, Tuple, Optional, Dict
import networkx as nx

_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2
_COLLISION_LAYERS = 2
_LAYER_AGENTS = 0
_LAYER_SHELFS = 1


class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, msg_bits: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


class Warehouse(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        n_agents: int,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
    ):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )

        self.n_agents = n_agents
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(sa_action_space)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []

        self.goals: List[Tuple[int, int]] = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

    def _is_highway(self, x: int, y: int) -> bool:
        return (
            (x % 3 == 0)
            or (y % 9 == 0)
            or (y == self.grid_size[0] - 1)
            or (
                (y > self.grid_size[0] - 11)
                and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
            )
        )

    def _make_obs(self, agent):

        y_scale, x_scale = self.grid_size[0] - 1, self.grid_size[1] - 1

        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]
        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)
        obs = {}
        obs["self"] = {
            "location": np.array([agent.x, agent.y]),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = self.agents[id_ - 1].message
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]
        return obs

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id
        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def reset(self):
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]
        agent_locs = np.random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.n_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.n_agents)
        self.agents = [
            Agent(x, y, dir_, self.msg_bits)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]
        self._recalc_grid()
        self.request_queue = list(
            np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )
        return tuple([self._make_obs(agent) for agent in self.agents])

    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict, bool]:
        assert len(actions) == len(self.agents)

        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)
        commited_agents = set()
        G = nx.DiGraph()
        for agent in self.agents:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            if (
                agent.carrying_shelf
                and start != target
                and self.grid[_LAYER_SHELFS, target[1], target[0]]
                and not (
                    self.grid[_LAYER_AGENTS, target[1], target[0]]
                    and self.agents[
                        self.grid[_LAYER_AGENTS, target[1], target[0]] - 1
                    ].carrying_shelf
                )
            ):
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(self.agents) - commited_agents
        for agent in failed_agents:
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP
        rewards = np.zeros(self.n_agents)
        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y
            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    agent.carrying_shelf = self.shelfs[shelf_id - 1]
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                if not self._is_highway(agent.x, agent.y):
                    agent.carrying_shelf = None
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        rewards[agent.id - 1] += 0.5
                    agent.has_delivered = False
        self._recalc_grid()
        shelf_delivered = False
        for x, y in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, y, x]
            if not shelf_id:
                if self.reward_type == RewardType.GLOBAL:
                    rewards -= 1
                elif self.reward_type == RewardType.INDIVIDUAL:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    rewards[agent_id - 1] -= 1
                elif self.reward_type == RewardType.TWO_STAGE:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    self.agents[agent_id - 1].has_delivered = True
                    rewards[agent_id - 1] -= 0.5
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                if self.reward_type == RewardType.GLOBAL:
                    rewards -= 1
                elif self.reward_type == RewardType.INDIVIDUAL:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    rewards[agent_id - 1] -= 1
                elif self.reward_type == RewardType.TWO_STAGE:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    self.agents[agent_id - 1].has_delivered = True
                    rewards[agent_id - 1] -= 0.5
                continue
            else:
                shelf_delivered = True
                new_request = np.random.choice(
                    list(set(self.shelfs) - set(self.request_queue))
                )
                self.request_queue[self.request_queue.index(shelf)] = new_request
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 1
                elif self.reward_type == RewardType.INDIVIDUAL:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    rewards[agent_id - 1] += 1
                elif self.reward_type == RewardType.TWO_STAGE:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    self.agents[agent_id - 1].has_delivered = True
                    rewards[agent_id - 1] += 0.5

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            dones = self.n_agents * [True]
        else:
            dones = self.n_agents * [False]
        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = {}
        return new_obs, list(rewards), dones, info, shelf_delivered

    def close(self):
        if self.renderer:
            self.renderer.close()


if __name__ == "__main__":
    shelf_rows = 1
    shelf_columns = 3
    n_agents = 1
    request_queue_size = int(n_agents * 2)
    reward_type = RewardType.GLOBAL
    msg_bits = 3
    max_inactivity_steps = None
    max_steps = 500
    sensor_range = 1
    column_height = 8

    env_wareHouse = Warehouse(
        shelf_columns,
        column_height,
        shelf_rows,
        n_agents,
        msg_bits,
        sensor_range,
        request_queue_size,
        max_inactivity_steps,
        max_steps,
        reward_type,
    )
    env_wareHouse.reset()
    for _ in range(100000):
        actions = env_wareHouse.action_space.sample()
        new_obs, rewards, dones, info, delivered = env_wareHouse.step(actions)
        print(rewards)
