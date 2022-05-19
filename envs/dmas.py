from typing import List, Tuple
import numpy as np
import cv2


UNKNOWN = 0
EMPTY = 1
VISITED = 2
WALL = 3
TARGET = 4
TARGET_FOUND = 5
AGENT = 6


COLOR_CODE = [  # GBR
    (20, 20, 20),  # almost black for unknow
    (220, 220, 220),  # almost white for empty
    (100, 100, 120),  # gray for visited
    (166, 84, 68),  # blue-ish for wall
    (0, 142, 171),  # gold for target
    (0, 47, 56),  # dark gold for target_found
    (115, 163, 109),  # green for agent
]


class DMAS:
    """
    discrete multui-agent search environment.
    class to handle multiagent movement in the environment,
    targets finding, and terminating when all targets are found.
    """

    def __init__(self,
                 grid: np.ndarray,  # 2d array of search map - padded with walls
                 n_agents: int,  # number of agents
                 n_targets: int,  # number of targets
                 ) -> None:
        self.actions = np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
        self.n_actions = len(self.actions)

        grid = grid.astype(int)
        self.grid = grid.copy()
        # grid which only contains targets
        self.target_grid = np.zeros_like(grid)

        self.empty_rows, self.empty_cols = np.where(
            grid == EMPTY)  # used for finding s0
        self.n_cells = len(self.empty_rows)  # number of empty cells

        self.n_agents = n_agents
        self.n_targets = n_targets
        self.targets_rows, self.targets_cols = None, None  # get assigned at reset

        # for visualization, get assigned at reset
        self.agents_rows, self.agents_cols = None, None

        self.frame_temp = np.zeros(
            (*grid.shape, 3), dtype=np.uint8)  # fir visualization

    def step(self,
             s: np.ndarray,  # agents positons [[r0, c1], ...]]
             a: List,  # list of action indices
             ) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        """receive environment state and agents actions, transition
        the agents positions, and terminate if all targets are found"""
        reward = np.zeros(self.n_agents) - 1

        sp = s + self.actions[a]

        # handle wall collisions - first get values of cells where the agents end up
        # then get the indices corresponding to walls, and keep those agents in place
        pos_vals = self.grid[sp[:, 0], sp[:, 1]]
        wall_clsns = np.where(pos_vals == WALL)[0]
        sp[wall_clsns] = s[wall_clsns]

        # if targets are found, mark them as found and reward agents
        target_vals = self.target_grid[sp[:, 0], sp[:, 1]]
        targets_found = np.where(target_vals == TARGET)[0]
        targets_indices = sp[targets_found]
        self.target_grid[targets_indices[:, 0],
                         targets_indices[:, 1]] = TARGET_FOUND

        # and reward agents
        reward[targets_found] = 1

        # handle done if all targets are found
        target_loc_vals = self.target_grid[self.targets_rows,
                                           self.targets_cols]
        done = np.all(target_loc_vals == TARGET_FOUND)

        # for visualization
        self.agents_rows, self.agents_cols = sp.T
        return sp, reward, done, {}

    def get_cv_frame(self, grid: np.ndarray) -> np.ndarray:
        """return a cv compatible gbr frame - h,w,c"""
        frame = self.frame_temp.copy()
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                color = COLOR_CODE[grid[i, j]]
                frame[i, j, :] = color
        return frame

    def render(self, size: Tuple[int, int] = (500, 500)) -> bool:
        grid = self.grid.copy()

        # draw agents
        grid[self.agents_rows, self.agents_cols] = AGENT

        # draw obstacles
        for i in range(len(self.targets_rows)):
            idx = (self.targets_rows[i], self.targets_cols[i])
            grid[idx] = self.target_grid[idx]

        # get cv compatible frame, resize and show it
        frame = self.get_cv_frame(grid)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        cv2.imshow("UML STRONG DISCRETE MULTI-AGENT SEARCH ENVIRONMENT", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return False
        return True

    def reset(self, n_agents: int = None, n_targets: int = None) -> np.ndarray:
        """reset env with potentially different number of agents and/or targets"""
        # handle n_agents and n_targets
        if n_agents is None:
            n_agents = self.n_agents
        else:
            self.n_agents = n_agents
        if n_targets is None:
            n_targets = self.n_targets
        else:
            self.n_targets = n_targets

        # get unique random initial positions for agents and targets
        indices = np.random.choice(
            len(self.empty_rows), size=n_agents + n_targets, replace=False)

        # place targets and save their locations
        self.targets_rows, self.targets_cols = self.empty_rows[indices[:n_targets]
                                                               ], self.empty_cols[indices[:n_targets]]
        self.target_grid[self.targets_rows, self.targets_cols] = TARGET

        # return agents positions as s
        s = np.array((self.empty_rows[indices[n_targets:]],
                     self.empty_cols[indices[n_targets:]])).T

        self.agents_rows, self.agents_cols = s.T
        self.frame_temp = np.zeros_like(self.frame_temp, dtype=np.uint8)
        return s


if __name__ == '__main__':
    n_agents = 100
    n_targets = 5
    grid_shape = (100, 100)
    window_shape = (700, 700)

    grid = np.random.choice((WALL, EMPTY), p=(0.1, 0.9), size=grid_shape)
    grid = np.pad(grid, (1, 1), mode="constant", constant_values=(WALL))
    env = DMAS(grid, n_agents, n_targets)

    s = env.reset()
    while True:
        a = np.random.choice(env.n_actions, n_agents)
        sp, _, done, _ = env.step(s, a)
        s = sp
        if not env.render(window_shape) or done:
            break
