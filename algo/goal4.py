import numpy as np
import torch
import networkx as nx
from utils.networks import Attnet
import collections
import itertools
import random


class Goal4:
    def __init__(self, env_params, args, action_nums=200):
        self.device = args.device
        self.type = args.type
        self.G = nx.DiGraph()
        self.parts = args.trajectory_part
        self.att = Attnet(env_params, hid_dim=128, out_dim=self.parts)
        self.att.to(self.device)
        self.certainty = [set() for _ in range(action_nums + 1)]
        self.cert_group = collections.defaultdict(int)
        self.optimizer = torch.optim.Adam(self.att.parameters(), lr=args.lr_goal)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def addgraph(self, ep_obs=None, ep_obs_d=None, ep_actions_d=None):
        # print(ep_obs,ep_obs_d,ep_actions_d)
        l = len(ep_obs_d)
        ep_obs_d = np.ravel(np.array(ep_obs_d))
        self.G.add_node(ep_obs_d[0], state=ep_obs[0])
        self.cert_group[ep_obs_d[0]] = 0
        for i in range(1, l):
            self.G.add_node(ep_obs_d[i], state=ep_obs[i])
            if ep_obs_d[i] not in self.cert_group:
                self.cert_group[ep_obs_d[i]] = 0
            self.G.add_edge(ep_obs_d[i - 1], ep_obs_d[i])
            if ep_obs_d[i - 1] not in self.certainty[ep_actions_d[i - 1]]:
                self.certainty[ep_actions_d[i - 1]].add(ep_obs_d[i - 1])
                self.cert_group[ep_obs_d[i - 1]] += 1

    def generategoal(self, curstate, curgoal, state_d, goal_d):
        ### aaaa upset, why can not study well aaaa delete delete give up 88
        batch_size = len(state_d)
        # state_d = list(_flatten(state_d))
        # goal_d = list(_flatten(goal_d))
        state_d = np.ravel(np.array(state_d))
        goal_d = np.ravel(np.array(goal_d))
        curstate = torch.tensor(curstate, dtype=torch.float32).to(self.device)
        curgoal = torch.tensor(curgoal, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            att = self.att(curstate, curgoal)
        category = torch.argmax(att, dim=1)
        newgoal = []

        if self.type == 'neighbor':
            group1 = [list(self.G.neighbors(i)) for i in state_d]
            group2 = [list(nx.single_source_shortest_path_length(self.G, i, cutoff=3)) for i in state_d]
            group3 = list(self.G.nodes)
            candidate = [group1, group2, group3]
            for i in range(batch_size):
                group = category[i].item()
                if group == 0 or group == 1:
                    idx = np.random.randint(0, len(candidate[group][i]))
                    node_idx = candidate[group][i][idx]
                else:
                    node_idx = random.choice(list(set(group3) - set(group1[i]) - set(group2[i])))
                newgoal.append(self.G.nodes[node_idx]['state'][:2])

        elif self.type == 'certainty':
            less_cert_group = collections.defaultdict(list)
            for k, v in self.cert_group.items():
                less_cert_group[v].append(k)
            certainty_nodes = [less_cert_group[k] for k in sorted(less_cert_group.keys())]
            certainty_nums = len(certainty_nodes)
            if certainty_nums>4:
                partitions = certainty_nums // 4
                group1 = certainty_nodes[0:partitions]
                group2 = certainty_nodes[partitions: partitions * 2]
                group3 = certainty_nodes[partitions * 2:partitions * 3]
                group4 = certainty_nodes[partitions * 3:certainty_nums]
                candidate = [group1, group2, group3, group4]
            else:
                candidate = [certainty_nodes,certainty_nodes,certainty_nodes, certainty_nodes]

            for i in range(batch_size):
                group = category[i].item()
                k = np.random.randint(0, len(candidate[group]))
                idx = np.random.randint(0, len(candidate[group][k]))
                node_idx = candidate[group][k][idx]
                newgoal.append(self.G.nodes[node_idx]['state'][:2])
        else:
            print('wrong graph partition ways')
            exit(0)

        return newgoal

    def update(self, training_samples):
        # not work orz
        curstate, curgoal, state_d, goal_d, desired_goals_d = training_samples
        batch_size = len(state_d)
        state_d = np.ravel(np.array(state_d))
        goal_d = np.ravel(np.array(goal_d))
        desired_goals_d = np.ravel(np.array(desired_goals_d))

        curstate = torch.tensor(curstate, dtype=torch.float32).to(self.device)
        curgoal = torch.tensor(curgoal, dtype=torch.float32).to(self.device)
        prediction = self.att(curstate, curgoal)
        y = []

        if self.type == 'neighbor':
            group1 = [list(self.G.neighbors(i)) for i in state_d]
            group2 = [list(nx.single_source_shortest_path_length(self.G, i, cutoff=3)) for i in state_d]
            group3 = list(self.G.nodes)
            #candidate = [group1, group2, group3]
            for i in range(batch_size):
                if desired_goals_d[i] in group1[i]:
                    y.append(0)
                elif desired_goals_d[i] in group2[i]:
                    y.append(1)
                else:
                    y.append(2)

        elif self.type == 'certainty':
            less_cert_group = collections.defaultdict(list)
            for k, v in self.cert_group.items():
                less_cert_group[v].append(k)
            certainty_nodes = [less_cert_group[k] for k in sorted(less_cert_group.keys())]
            certainty_nums = len(certainty_nodes)
            if certainty_nums>3:
                partitions = certainty_nums // 4
                group1 = certainty_nodes[0:partitions]
                group2 = certainty_nodes[partitions: partitions * 2]
                group3 = certainty_nodes[partitions * 2:partitions * 3]
                group4 = certainty_nodes[partitions * 3:certainty_nums]
                candidate = [list(itertools.chain.from_iterable(group1)), list(itertools.chain.from_iterable(group2)), list(itertools.chain.from_iterable(group3)), list(itertools.chain.from_iterable(group4))]
            else:
                candidate = [list(itertools.chain.from_iterable(certainty_nodes))]*3
            for i in range(batch_size):
                if desired_goals_d[i] in candidate[0]:
                    y.append(0)
                elif desired_goals_d[i] in candidate[1]:
                    y.append(1)
                elif desired_goals_d[i] in candidate[2]:
                    y.append(2)
                else:
                    y.append(3)
        else:
            print('wrong graph partition ways')
            exit(0)

        y = torch.tensor(y, dtype=torch.long).to(self.device)
        loss = self.loss_func(prediction, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
