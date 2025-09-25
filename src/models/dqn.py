import math
import torch
import torch.nn as nn
from torch.distributions import Categorical, RelaxedOneHotCategorical, Bernoulli
from torch.distributions.normal import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy, AUROC
import os
from accelerate import Accelerator

from src.utils.utils import MLP

from tqdm.auto import tqdm

GAMMA = 0.9  # Discount factor
LAMBDA = 0.95  # GAE parameter
BETA_1 = 0.01
BETA_2 = 0.001
EPSILON = 0.3

def valid_probs(preds):
    '''Ensure valid probabilities.'''
    return torch.all((preds >= 0) & (preds <= 1))

def calculate_criterion(preds, task):
    '''Calculate feature selection criterion.'''
    if task == 'regression':
        # Calculate criterion: prediction variance.
        return torch.var(preds)

    elif task == 'classification':
        if not valid_probs(preds):
            preds = preds.softmax(dim=1)

        mean = torch.mean(preds, dim=0, keepdim=True) # Take mean prediction across MC samples
        kl = torch.sum(preds * torch.log(preds / (mean + 1e-6) + 1e-6), dim=1) # KL divergence between individual MC prediction and mean prediction
        return torch.mean(kl) # Expectation of KL divergence over MC samples
    else:
        raise ValueError(f'unsupported task: {task}. Must be classification or regression')
    

class DuelingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, dropout=0):
        super(DuelingNet, self).__init__()
        self.pi_net = MLP(input_dim, n_actions, hidden_dim, 2, dropout)
        self.v_net = MLP(input_dim, 1, hidden_dim, 2, dropout)
        self.n_actions = n_actions

    def forward(self, x, m):
        x = x * m
        inputs = torch.cat([x, m], dim=-1)
        self.adv = self.pi_net(inputs) # batch_size x n_actions
        self.v = self.v_net(inputs) # batch_size x 1
        output = self.v + (self.adv - torch.mean(self.adv, dim=1, keepdim=True))

        return output
    
class DQNetwork(nn.Module):
    def __init__(self, qnet, predictor, config):
        super(DQNetwork, self).__init__()
        self.qnet = qnet
        self.predictor = predictor
        self.task = config['task']
        self.dim_y = config['label_dim']

    def forward(self, x, m, actions):
        q_vals = self.qnet(x, m)
        return q_vals[torch.arange(x.shape[0]), actions]

    def act(self, x, m, available, eps=None, deterministic=False):
        q_val = self.qnet(x, m)
        N, n_actions = q_val.size()

        exploration_prob = torch.ones(N, n_actions, out=q_val.new())

        # Set nonavailable actions to not be acquirable
        if not available.all():
            ind = torch.nonzero(1-available)
            q_val[ind[:, 0], ind[:, 1]] = -np.inf
            exploration_prob[ind[:, 0], ind[:, 1]] = 0

        valid_rows = available.sum(dim=1) > 0
        action = torch.full((N,), -1, dtype=torch.long, device=q_val.device)

        if valid_rows.any():
            # For valid rows, choose the max Q-value among available actions
            max_q_val, best_action = q_val[valid_rows].max(dim=1)
            action[valid_rows] = best_action

            if eps is not None:
                noise = q_val.new(N).uniform_()
                exploration = (noise < eps) & valid_rows

                if exploration.any():
                    rand_rows = torch.nonzero(exploration).squeeze()
                    random_action = torch.multinomial(
                        exploration_prob[rand_rows], 1, replacement=True
                    ).squeeze()
                    action[rand_rows] = random_action

        return action

    def predict(self, x, m):
        x = x * m
        x = torch.cat([x, m], dim=-1)
        return self.predictor(x)
    
    def evaluate(self, x, m, y):
        x = x * m
        x = torch.cat([x, m], dim=-1)

        if self.task == 'regression':
            mean, std = self.predictor(x)
            dist = Normal(mean.view(-1), std.view(-1))
            log_probs = dist.log_prob(y)
        elif self.task == 'classification':
            logits = self.predictor(x)
            if self.dim_y == 1:
                probs = torch.sigmoid(logits.view(-1))
                dist = Bernoulli(probs)
                log_probs = dist.log_prob(y)
            else:
                probs = torch.softmax(logits, dim=-1)
                targets = y.argmax(dim=-1)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(targets)

        return dist, log_probs

# Buffer for storing rollouts from current actor and critic
class RolloutBuffer:
    def __init__(self):
        self.x = []
        self.m = []
        self.m_next = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.labels = []
        self.exists = []

    def add(self, x, m, m_next, action, reward, done, labels, exists):
        self.x.append(x)
        self.m.append(m)
        self.m_next.append(m_next)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.labels.append(labels)
        self.exists.append(exists)

    def clear(self):
        self.x.clear()
        self.m.clear()
        self.m_next.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.labels.clear()
        self.exists.clear()

    def get_batch(self):
        mb_x = torch.stack(self.x).view(-1, self.x[0].shape[1])
        mb_m = torch.stack(self.m).view(-1, self.m[0].shape[1])
        mb_m_next = torch.stack(self.m_next).view(-1, self.m[0].shape[1])
        mb_exist = torch.stack(self.exists).view(-1, self.exists[0].shape[1])
        mb_labels = torch.stack(self.labels).view(-1)
        mb_actions = torch.stack(self.actions).view(-1)

        return mb_x, mb_m, mb_m_next, mb_exist, mb_labels, mb_actions

class Env(object):
    def __init__(self, config, data, device='cuda'):
        self.dataset = data
        self.device = device

        self.n_data = len(self.dataset)
        self.task = config['task']
        self.dim_y = config['label_dim']
        self.n_features = config['feature_dim']
        self.free_indices = config['free_indices']
        self.num_available = len([i for i in np.arange(self.n_features) if i not in self.free_indices])

        self.batch_inputs, self.batch_exist, self.batch_labels = self.dataset.tensors
        self.batch_inputs = self.batch_inputs.to(self.device)
        self.batch_exist = self.batch_exist.to(self.device)

        self.reset()

    def reset(self):
        self.batch_acquired = torch.zeros((self.n_data, self.n_features), dtype=torch.uint8, device=self.device)
        self.action_count = torch.zeros((self.n_data))

        if self.free_indices is not None:
            self.batch_acquired[:, self.free_indices] = 1

    def set_model(self, model):
        self.model = model

    def mask_inputs(self, x, m):
        x_input = torch.cat([x * m, m], dim=-1)
        return x_input

    def get_initial_data(self):
        m = torch.zeros_like(self.batch_inputs)
        if self.free_indices is not None:
            m[:, self.free_indices] = 1
        return self.batch_inputs, m

    def calc_reward(self, logits, labels):
        if self.task == 'classification':
            if self.dim_y == 1:
                loss = F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1), reduction='none')
                reward = -loss
                prob = logits.sigmoid()
            else:
                targets = labels.argmax(dim=-1)  # [B, T]
                dist = Categorical(logits=logits)
                reward = dist.log_prob(targets).unsqueeze(-1)
                prob = dist.probs

        return reward, prob

    def _nonterminal_step(self, actions, nonterminal):
        self.batch_acquired[nonterminal, actions[nonterminal]] = 1
        self.action_count[nonterminal] += 1

    def step(self, actions):
        # For samples that are not done, perform action and update obs
        # For samples that are done, perform classification and get reward
        nonterminal = np.array((actions != -1))
        rewards = torch.zeros(self.n_data)

        if np.any(nonterminal):
            self._nonterminal_step(actions, nonterminal)

        done = (self.action_count == self.num_available) | (~nonterminal)
        done = done.to(torch.bool)

        # probs = torch.zeros(self.n_data)
        # if torch.any(done):
        #     p_y_logit = self.model.predict(self.batch_inputs[done], self.batch_acquired[done]).detach().cpu()
        #     reward_, prob_ = self.calc_reward(p_y_logit, self.batch_labels[done])
        #     rewards[done] = reward_.squeeze(-1)
        #     probs[done] = prob_.squeeze(-1)

        p_y_logit = self.model.predict(self.batch_inputs, self.batch_acquired).detach().cpu()
        rewards, probs = self.calc_reward(p_y_logit, self.batch_labels)
        rewards = rewards.squeeze(-1)
        probs = probs.squeeze(-1)

        return self.batch_acquired, rewards, done, probs

# Generate history from current policy
def generate_rollouts(env, buffer, model, eps):
    env.reset()
    env.set_model(model)
    x, m = env.get_initial_data()

    done = torch.zeros(x.shape[0], dtype=torch.bool, device='cpu')

    with torch.no_grad():
        while not torch.all(done):
            exist = env.batch_exist
            acquired = env.batch_acquired
            available = exist * (1 - acquired)

            action = model.act(x, m, available, eps, False)
            next_m, reward, done, _ = env.step(action.cpu())

            labels = env.batch_labels 
            exists = env.batch_exist
            done_int = done.to(dtype=torch.int)
            buffer.add(x, m, next_m, action, reward, done_int, labels, exists)
            m = next_m
        
    return buffer

# def compute_returns(rewards, dones, last_value=0, gamma=GAMMA, lam=LAMBDA):
#     returns = []
#     discounted_reward = last_value
#     for i in reversed(range(len(rewards))):
#         reward = rewards[i].squeeze()
#         done = dones[i].squeeze()
        
#         discounted_reward = reward + gamma * discounted_reward * (1 - done)

#         # print(done[0:5])
#         # print(reward[0:5])
#         # print(discounted_reward[0:5])

#         returns.insert(0, discounted_reward)

#     return returns

def compute_td_targets(dqnet_old, x, next_m, rewards, dones, gamma=0.9):
    with torch.no_grad():
        # Q-values for next state from target net
        q_next = dqnet_old.qnet(x, next_m)   # shape [B, num_actions]
        max_q_next, _ = q_next.max(dim=1)      # greedy action value

        # Bellman target
        targets = rewards + gamma * (1 - dones) * max_q_next
    return targets

def test(env, model):
    env.reset()
    env.set_model(model)
    x, m = env.get_initial_data()
    done = torch.zeros(env.n_data)
    returns = torch.zeros(env.n_data)

    classified = torch.zeros(x.shape[0], dtype=torch.bool, device='cpu')

    with torch.no_grad():
        while not torch.all(done):
            acquired_ = env.batch_acquired
            exist_ = env.batch_exist
            labels_ = env.batch_labels

            available_ = exist_ * (1 - acquired_)
            action = model.act(x, m, available_, 0, False)

            action[classified] = -1
            next_m, reward_, done, probs_ = env.step(action.cpu())

            returns += reward_
            m = next_m
    
    return torch.mean(returns)

def get_outputs(config, env, model):
    outputs_dict = {
        'features': {},
        'acquisitions': {},
        'preds': {},
        'missingness': {},
        'labels': {},
        'indices': {}
    }

    score_dict = {num: None for num in range(config['num_available_actions'])}

    env.set_model(model)
    x, m = env.get_initial_data()

    model.eval()

    for num in range(config['num_available_actions']):
        acquired_ = env.batch_acquired
        exist_ = env.batch_exist
        labels_ = env.batch_labels

        available_ = exist_ * (1 - acquired_)
        action = model.act(x, m, available_, 0, True)
        next_x, next_m, _, _, _ = env.step(action.cpu())
        p_y_logit = model.predict(next_x, next_m).detach().cpu()
        probs_ = p_y_logit.sigmoid()

        auc = AUROC(task='multiclass', num_classes=probs_.shape[1])(probs_, labels_.squeeze())

        score_dict[num] = auc.item()

        outputs_dict['preds'][num] = probs_
        outputs_dict['acquisitions'][num] = env.batch_acquired.float()
        outputs_dict['labels'][num] = labels_.squeeze()
        outputs_dict['features'][num] = env.batch_inputs.cpu()

        m = next_m

    return score_dict, outputs_dict


def test_and_record_dqn(
        test_dataset,  
        model,
        config):
    
    testenv = Env(config, test_dataset)

    testenv.reset()
    score_dict, outputs_dict = get_outputs(config, testenv, model)
    
    return score_dict, outputs_dict


def train_dqn(train_dataset,
              val_dataset,
                predictor,
                config):
    
    qnet = DuelingNet(config['feature_dim'] * 2, config['nn_hidden_dim'], config['feature_dim'])
    dqnet = DQNetwork(qnet=qnet, predictor=predictor, config=config)
    
    dqnet_old = type(dqnet)(
        qnet=deepcopy(dqnet.qnet),
        predictor=deepcopy(dqnet.predictor),
        config=config
    )
    dqnet_old.load_state_dict(dqnet.state_dict())

    optimizer = optim.Adam(dqnet.qnet.parameters(), lr=config['lr_rl'])
    accelerator = Accelerator(mixed_precision="bf16")
    optimizer, dqnet_old, dqnet = accelerator.prepare(optimizer, dqnet_old, dqnet)

    device = accelerator.device

    env = Env(config, train_dataset, device=device)
    valenv = Env(config, val_dataset, device=device)
    
    buffer = RolloutBuffer()

    best_score = float('-inf')
    best_qnet = None 
    episodes = config['episodes']
    buffer_size = config['buffer_size']
    batch_size = config['batch_size_rl']
    verbose = config['verbose']
    inner_loop_epoch = config['inner_loop_steps_rl']
    target_update = config['target_update']

    eps = 1
    eps_decay = np.exp(np.log(0.1) / episodes)
    eps_min = 0.5

    for episode in range(episodes): 
        buffer.clear()
        x, m, m_next, actions, rewards, labels, dones = [], [], [], [], [], [], []
        len_buffer = 0
        
        while len_buffer < buffer_size:
            buffer.clear()
            buffer = generate_rollouts(env, buffer, dqnet_old, eps)

            reward = buffer.rewards
            done = buffer.dones

            mb_x, mb_m, mb_next_m, _, mb_labels, mb_actions = buffer.get_batch()
            mb_actions = mb_actions.to(device)
            mb_labels = mb_labels.to(device)
            mb_done = torch.stack(done).view(-1).to(device)
            mb_rewards = torch.stack(reward).view(-1).to(device)

            # returns_ = compute_returns(rewards, dones)
            # mb_returns = torch.stack(returns_).view(-1).to(device)

            x.append(mb_x)
            m.append(mb_m)
            m_next.append(mb_next_m)
            actions.append(mb_actions)
            rewards.append(mb_rewards)
            labels.append(mb_labels)
            dones.append(mb_done)

            # Monitor size of buffer
            len_buffer += mb_x.shape[0]

        x = torch.cat(x, dim=0)
        m = torch.cat(m, dim=0)
        m_next = torch.cat(m_next, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        labels = torch.cat(labels, dim=0)
        dones = torch.cat(dones, dim=0)

        # Reward normalization
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        rewards = (rewards - mean_r) / std_r

        returns = compute_td_targets(dqnet_old, x, m_next, rewards, dones)

        # Train 
        dqnet.train()
        sampler = BatchSampler(SubsetRandomSampler(
                range(x.shape[0])), batch_size, drop_last=True)
        
        train_loss = 0

        for _ in range(inner_loop_epoch):
            for indices in sampler:
                x_ = x[indices]
                m_ = m[indices]
                actions_ = actions[indices]
                returns_ = returns[indices]

                q_vals_ = dqnet(x_, m_, actions_)
                # Compute loss
                loss = nn.MSELoss()(q_vals_, returns_)
                train_loss += loss.detach().cpu().item()

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(dqnet.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

        if episode % target_update == 0:
            dqnet_old.load_state_dict(dqnet.state_dict())

        dqnet.eval()
        reward = test(valenv, dqnet)

        if episode % 10 == 0:
            if verbose:
                print(f"Episode: {episode}: {reward}")

        if reward > best_score:
            if verbose:
                print("Model saved")
            best_score = reward
            best_qnet = dqnet.state_dict()

        eps = max(eps_min, eps_decay * eps)

    if best_qnet is not None:
        dqnet.load_state_dict(best_qnet)

    return dqnet
