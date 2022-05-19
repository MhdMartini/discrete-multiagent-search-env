# TODO: Change the trust decisions from a trust matrix to trust actions by the agents
from typing import Tuple
from dmas import AGENT, EMPTY, WALL, DMAS
import numpy as np
from utils.env_utils import fast_fov, get_distances, get_comm_dist_mat
from itertools import count
import cv2


# for visualization
FRAME = 0
SELF_COLOR = (0, 10, 120)


class DMASI:
    def __init__(self,
                 env: DMAS,  # dmas environment
                 fovs: np.ndarray,  # agents fovs
                 comm_ranges: np.ndarray,  # agents' communication ranges
                 trust_matrix: np.ndarray = None,  # NxN relational matrix
                 ) -> None:
        self.env = env
        self.shape_2d = env.grid.shape

        self.masks = None
        self.fovs = fovs
        self.dmas_s = None  # state of the dmas env (agents' positions vector)

        self.comm_ranges = comm_ranges
        self.comm_dist_mat = get_comm_dist_mat(
            comm_ranges.reshape(-1, 1))  # communication distance matrix
        self.comm_matrix = None  # agents within communication distance and want to communicate

        # store trust network. if None, store identity matrix
        self.trust_matrix = trust_matrix if trust_matrix is not None else np.ones(
            (env.n_agents, env.n_agents))
        self.agents_indices = range(env.n_agents)

    def step(self, _, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        1- step the dmas environment
        2- update agents' masks according to the new positions
        3- check for agents in the vicinity of each other; 
        share masks according to trust
        4- generate a pre-state, then apply the masks to it
        5- and return it
        """
        self.dmas_s, reward, done, info = self.env.step(self.dmas_s, a)
        self.masks = self.step_masks(self.masks, self.dmas_s, self.fovs)
        sp = self.construct_s(self.masks)
        return sp, reward, done, info

    def get_trust_actions_matrix(self) -> np.ndarray:
        return np.random.binomial(1, self.trust_matrix)

    def step_masks(self, masks: np.ndarray, positions: np.ndarray, fovs: np.ndarray) -> np.ndarray:
        """step the environment with respect to masks"""
        # update around the agents according to fovs
        masks = self.update_masks(masks, positions, fovs)

        # get communication matrix
        self.comm_matrix = self.get_comm_matrix(positions)

        # share masks according to trust matrix
        masks = self.share_masks(masks, self.comm_matrix)
        return masks

    def update_masks(self, masks: np.ndarray, positions: np.ndarray, fovs: np.ndarray) -> np.array:
        """set the new locations around the agents to 1. each agent
        could have a unique fov"""
        for agent_id in range(self.env.n_agents):
            masks[agent_id] = fast_fov(
                masks[agent_id], positions[agent_id], fovs[agent_id])
        return masks

    def get_comm_matrix(self, positions: np.ndarray) -> np.ndarray:
        """get matrix which is True at indices where agets are within 
        communication distance to one another and decide to integrate with
        one another"""
        dist_mat = get_distances(positions)
        within_comm = dist_mat <= self.comm_dist_mat
        trust_actions_matrix = self.get_trust_actions_matrix()
        comm_matrix = np.logical_and(within_comm, trust_actions_matrix)
        return comm_matrix

    def share_masks(self, masks: np.ndarray, comm_matrix: np.ndarray) -> np.ndarray:
        """Loop through the agents and share masks according to distance 
        and  trust matrix"""
        for i in range(comm_matrix.shape[0]):
            for j in range(comm_matrix.shape[1]):
                if not comm_matrix[i, j] or i == j:
                    # continue if it's the same agent
                    continue
                masks[i] = np.logical_or(masks[i], masks[j])
        return masks

    def construct_s(self, masks: np.ndarray) -> np.ndarray:
        """update the ground-truth state according to agents positions
        and targets status - return a 3d image of environment"""
        grid = self.env.grid.copy()

        # place targets
        targets_indices = (self.env.targets_rows, self.env.targets_cols)
        grid[targets_indices] = self.env.target_grid[targets_indices]

        # turn into image and return it
        pre_s = self.env.get_cv_frame(grid)
        s = pre_s * masks[:, ..., None]  # apply mask
        s = self.color_agents(s, self.comm_matrix)
        return s

    def color_agents(self, s: np.ndarray, comm_matrix: np.ndarray) -> np.ndarray:
        """color agents on their frame, in addtion to others
        who share info with"""
        alice, bob = np.where(comm_matrix)
        s[alice, self.dmas_s[bob, 0],
          self.dmas_s[bob, 1], :] = AGENT
        s[self.agents_indices, self.dmas_s[self.agents_indices, 0],
          self.dmas_s[self.agents_indices, 1], :] = SELF_COLOR
        return s

    def render(self, size: Tuple[int, int] = (700, 700)) -> bool:
        global FRAME

        # check if user quits
        key = cv2.waitKey(35)
        if key == ord('q'):
            return False

        if key == ord('g'):
            FRAME = 0
            cv2.destroyAllWindows()

        # check if user toggles
        elif key == ord(' '):
            FRAME = (FRAME + 1) % (self.env.n_agents + 1)
            cv2.destroyAllWindows()

        if FRAME == 0:
            # render base env instead
            self.env.render(size=size)
            return True

        s = self.construct_s(self.masks)
        frame = cv2.resize(s[FRAME - 1], size,
                           interpolation=cv2.INTER_AREA)

        cv2.imshow(
            f"Agent {FRAME - 1}: FOV = {self.fovs[FRAME - 1]}, Range = {self.comm_ranges[FRAME - 1]}", frame)

        return True

    def reset(self) -> np.ndarray:
        """
        return a 3d array with a grid for each agent. each agent's 
        grid is made of its mask applied to the ground truth image
        1. reset the dmas environment and store agents' initial positions vector
        2. reset agents masks according to their fov's and positions
        3. reset gt_s image according to agents' positions
        4. generate each state's channel by applying each mask to gt_s
        5. return those states as s.
        """
        self.dmas_s = self.env.reset()
        self.masks = np.zeros((self.env.n_agents, *self.shape_2d), dtype=bool)
        self.comm_matrix = np.eye(self.env.n_agents, dtype=bool)
        s = self.construct_s(self.masks)
        return s


if __name__ == '__main__':
    n_agents = 100
    n_targets = 5
    grid_shape = (100, 100)
    window_shape = (700, 700)

    grid = np.random.choice((WALL, EMPTY), p=(0.1, 0.9), size=grid_shape)
    grid = np.pad(grid, (1, 1), mode="constant", constant_values=(WALL))
    env_m = DMAS(grid, n_agents, n_targets)

    fovs = np.random.randint(1, 7, n_agents, dtype=int)
    comm_ranges = np.random.randint(1, 7, n_agents, dtype=int)
    env = DMASI(env_m, fovs, comm_ranges)

    s = env.reset()
    for i in count():
        a = np.random.choice(env_m.n_actions, n_agents)
        sp, _, done, _ = env.step(s, a)
        s = sp
        if not env.render(window_shape) or done:
            break
