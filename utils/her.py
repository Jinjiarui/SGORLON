import numpy as np


class HER:
    def __init__(self, replay_strategy, replay_k, threshold, goal=None, future_step=200):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.future_p = 1 - (1. / (1 + replay_k))
        # if self.replay_strategy == 'future':
        #     self.future_p = 1 - (1. / (1 + replay_k))
        # else:
        #     self.future_p = 0
        self.threshold = threshold
        self.goal_sampler = goal
        self.future_step = future_step

    def reward_func(self, state, goal, info=None):
        assert state.shape == goal.shape
        dist = np.linalg.norm(state - goal, axis=-1)
        return -(dist > self.threshold).astype(np.float32)

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, graph_warmup=0):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        if self.replay_strategy == 'future' or not graph_warmup:
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            # replace go with achieved goal
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag
        elif self.replay_strategy == 'graph' and graph_warmup:
            curstate, curgoal= transitions['obs'][her_indexes], transitions['ag'][her_indexes]
            state_d, goal_d = transitions['obs_d'][her_indexes], transitions['ag_d'][her_indexes]
            future_ag = self.goal_sampler.generategoal(curstate, curgoal, state_d, goal_d)
            transitions['g'][her_indexes] = future_ag
        else:
            print('wrong replay_strategy')
            exit(0)
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions

    def sample_goal_training(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        target_index = np.minimum(T, t_samples + self.future_step)
        future_offset = np.random.uniform(size=batch_size) * (target_index - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        curstate, curgoal = transitions['obs'][her_indexes], transitions['g'][her_indexes]
        state_d, goal_d = transitions['obs_d'][her_indexes], transitions['ag_d'][her_indexes]
        ###planning not done
        desired_goals_d = episode_batch['ag_d'][episode_idxs[her_indexes], future_t]
        training_samples = [curstate, curgoal, state_d, goal_d, desired_goals_d]
        return training_samples