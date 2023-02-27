import numpy as np

"""
the replay buffer here is basically from the openai baselines code
"""


class ReplayBuffer:
    def __init__(self, env_params, buffer_size, warmup, sample_func, goal_sample_func=None):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.warmup = warmup
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        self.goal_sample_func = goal_sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        'r':np.empty([self.size, self.T, 1]),
                        'obs_d':np.empty([self.size, self.T+1, 1]),
                        'ag_d': np.empty([self.size, self.T + 1, 1])
                        }

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, mb_r, mb_obs_d, mb_ag_d = episode_batch
        batch_size = mb_obs.shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.buffers['r'][idxs] = mb_r
        self.buffers['obs_d'][idxs] = mb_obs_d
        self.buffers['ag_d'][idxs] = mb_ag_d
        self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        if self.current_size>self.warmup:
            transitions = self.sample_func(temp_buffers, batch_size, 1)
        else:
            transitions = self.sample_func(temp_buffers, batch_size, 0)
        return transitions

    def sample_goal_training(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        training_samples = self.goal_sample_func(temp_buffers, batch_size)
        return training_samples

    def random_sample(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        #print('start random sample', self.current_size)
        T = temp_buffers['actions'].shape[1]  # 50 steps per traj
        rollout_batch_size = temp_buffers['actions'].shape[0]  # 2 trajs
        batch_size = batch_size  # target batches we want to sample
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # which traj to sample
        t_samples = np.random.randint(T, size=batch_size)
        # which step to sample
        transitions = {key: temp_buffers[key][episode_idxs, t_samples].copy() for key in temp_buffers.keys()}
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        return transitions

    def sample_traj(self, batch_size):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        T = temp_buffers['actions'].shape[1]  # 50 steps per traj
        num_traj = temp_buffers['actions'].shape[0] #number of all the trajs
        episode_idxs = np.random.randint(0, num_traj, batch_size)
        traj = {key: temp_buffers[key][episode_idxs, :].copy() for key in temp_buffers.keys()}
        return traj

    def get_all_data(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        return temp_buffers

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx