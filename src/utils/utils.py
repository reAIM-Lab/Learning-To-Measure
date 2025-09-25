import logging

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, RelaxedOneHotCategorical, Categorical, Bernoulli
from torch.utils.data import TensorDataset, DataLoader, random_split

import pytorch_lightning as pl
# import xgboost as xgb

def get_logger(logfilename):
    logger = logging.getLogger("sim_logger")
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers if get_logger is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(logfilename)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def generate_uniform_mask(batch_size, num_features, seed=None):
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        unif = torch.rand(batch_size, num_features, generator=generator)
        ref = torch.rand(batch_size, 1, generator=generator)
    else:
        unif = torch.rand(batch_size, num_features)
        ref = torch.rand(batch_size, 1)
    mask = (unif > ref).float()
    return mask

def make_onehot(x):
    '''Make an approximately one-hot vector one-hot.'''
    argmax = torch.argmax(x, dim=-1)  # Get argmax along last dimension
    batch_size = x.shape[0]

    if len(x.shape) == 2:
        onehot = torch.zeros_like(x)
        # For each batch and sequence position, set the argmax index to 1
        onehot[torch.arange(batch_size), argmax] = 1
        return onehot
    elif len(x.shape) == 3:
        seq_len = x.shape[1]
        onehot = torch.zeros_like(x)
        # For each batch and sequence position, set the argmax index to 1
        onehot[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), argmax] = 1
        return onehot
    else:
        raise ValueError(f"Unsupported shape: {x.shape}")

class ConcreteSelector(nn.Module):
    '''Output layer for selector models.'''

    def __init__(self):
        super().__init__()

    def forward(self, logits, temp, deterministic=False, feature_groups=None, available=None):
        if feature_groups is not None:        
            if available is not None:
                logits = logits * available

            group_logits = []
            for indices in feature_groups.values():
            
                if available is not None:
                    group_available = available[:, :, indices]  # Shape: (batch_size, seq_len, group_size)
                    all_unavailable = (group_available.sum(dim=-1) == 0)  # Shape: (batch_size, seq_len)
                    group_logits.append(torch.where(all_unavailable, 
                                                    logits[:, :, indices].sum(dim=-1) - 1e8, 
                                                    logits[:, :, indices].sum(dim=-1)))
                else:
                    group_logits.append(logits[:, :, indices].sum(dim=-1))   # Sum logits per group

            group_logits = torch.stack(group_logits, dim=-1)  # Shape: (batch_size, num_groups)
        else:
            if available is not None:
                logits = logits - 1e8 * (1 - available)
            group_logits = logits

        # If deterministic, compute the argmax. If not, sample from softmax distribution.
        if deterministic:
            probs = torch.softmax(group_logits / temp, dim=-1)
            sample = F.one_hot(probs.argmax(dim=-1), num_classes=probs.size(-1)).float()
        else:
            dist = RelaxedOneHotCategorical(temp, logits=group_logits)
            soft_sample = dist.rsample()

            hard_sample = torch.zeros_like(soft_sample)
            hard_sample.scatter_(-1, soft_sample.argmax(dim=-1, keepdim=True), 1.0)
            sample = hard_sample.detach() - soft_sample.detach() + soft_sample

        if feature_groups is not None:
            batch_size, seq_len, _ = logits.shape
            num_features = logits.size(-1)
            mapped_sample = torch.zeros((batch_size, seq_len, num_features), device=logits.device)

            for group_idx, indices in enumerate(feature_groups.values()):
                mapped_sample[:, :, indices] = sample[:, :, group_idx].unsqueeze(-1)

            return mapped_sample  # (B, T, F)
        else:
            return sample 
        
def MLP(input_dim, output_dim, hidden_dim, num_layers, policy_head=False, dropout=None):
    layers = []
    if num_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())

            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))

    if policy_head:
        nn.init.orthogonal_(layers[-1].weight, gain=0.01)
        nn.init.constant_(layers[-1].bias, 0.0)

    return nn.Sequential(*layers)

class BaselineNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='regression', dropout=0):
        super().__init__()
        self.task = task

        if self.task == 'regression':
            self.mlp = MLP(input_dim, output_dim * 2, hidden_dim, 2, dropout)
        elif self.task == 'classification':
            self.mlp = MLP(input_dim, output_dim, hidden_dim, 2, dropout)

    def forward(self, x):
        out = self.mlp(x)

        if self.task == 'regression':
            mean, log_std = torch.chunk(out, 2, dim=-1)
            std = 0.05 + 0.95 * F.softplus(log_std)
            return mean, std
        elif self.task == 'classification':
            return out
    

class BaselineNNLightning(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=5e-4, task='regression', dropout=0, mask_random=True, free_indices=None):
        super().__init__()
        self.save_hyperparameters()
        self.feature_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.mask_random = mask_random
        self.task = task
        self.model = BaselineNN(input_dim * 2, hidden_dim, output_dim, task=task, dropout=dropout)
        self.free_indices = free_indices

    def forward(self, x, m):
        x = x * m
        x = torch.cat([x, m], dim=-1)
        return self.model(x)
    
    def mask_features(self, x, r, seed=None):
        """ Randomly masks features with probability `mask_prob` by setting them to zero """
        if self.mask_random:
            mask = generate_uniform_mask(x.shape[0], x.shape[1], seed=seed).to(x.device)
        else:
            mask = torch.ones_like(x)
        
        mask = mask * r 
        if self.free_indices is not None:
            mask[:, self.free_indices] = 1
        
        return mask

    def training_step(self, batch, batch_idx):
        x, r, y = batch
        m = self.mask_features(x, r)

        if self.task == 'regression':
            mean, std = self(x, m)
            mean = mean.view(-1)
            std = std.view(-1)

            loss = -Normal(mean, std).log_prob(y).mean()
        elif self.task == 'classification':
            logits = self(x, m)
            if self.output_dim == 1:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), y)
            else:
                targets = y.argmax(dim=-1)  # [B, T]
                dist = Categorical(logits=logits)
                loss = -dist.log_prob(targets).mean()

        self.log("train_loss", loss.detach().item(), prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, r, y = batch
        m = self.mask_features(x, r, seed=42)

        if self.task == 'regression':
            mean, std = self(x, m)
            mean = mean.view(-1)
            std  = std.view(-1)

            loss = -Normal(mean, std).log_prob(y).mean()
        elif self.task == 'classification':
            logits = self(x, m)
            if self.output_dim == 1:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), y)
            else:
                targets = y.argmax(dim=-1)  # [B, T]
                dist = Categorical(logits=logits)
                loss = -dist.log_prob(targets).mean()

        self.log("val_loss", loss.detach().item(), prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def evaluate(self, x, m, y=None):
        with torch.no_grad():
            if self.task == 'regression':
                mean, std = self(x, m)
                dist = Normal(mean.view(-1), std.view(-1))
                log_probs = dist.log_prob(y)
                return dist, log_probs

            elif self.task == 'classification':
                logits = self(x, m)
                if self.output_dim == 1:
                    probs = torch.sigmoid(logits.view(-1))
                    dist = Bernoulli(probs)
                    log_probs = dist.log_prob(y)
                    return dist, log_probs
                else:
                    probs = torch.softmax(logits, dim=-1)
                    targets = y.argmax(dim=-1)
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(targets)
                    return dist, log_probs
                    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
    
def get_loaders(X, R, Y, config):
    full_dataset = TensorDataset(X, R, Y)
    val_len = max(1, int(config['val_frac_baseline'] * len(full_dataset)))
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_baseline'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False, pin_memory=True)

    return train_loader, val_loader

def get_imputed_datasets(X, R, Y, config):
    feature_sums = (X * R).sum(dim=0)         # sum over observed entries
    feature_counts = R.sum(dim=0).clamp(min=1)  # avoid divide by zero
    feature_means = feature_sums / feature_counts  # [D]

    # Broadcast replace missing values with means
    X_imputed = X.clone()
    for d in range(X.shape[1]):
        X_imputed[:, d][R[:, d] == 0] = feature_means[d]

    R = torch.ones_like(X_imputed)

    N = len(X_imputed)
    val_len = max(1, int(config['val_frac_baseline'] * N))
    train_len = N - val_len
    indices = torch.randperm(N)

    train_idx = indices[:train_len]
    val_idx = indices[train_len:]

    X_train, R_train, Y_train = X_imputed[train_idx], R[train_idx], Y[train_idx]
    X_val, R_val, Y_val = X_imputed[val_idx], R[val_idx], Y[val_idx]

    train_dataset = TensorDataset(X_train, R_train, Y_train)
    val_dataset = TensorDataset(X_val, R_val, Y_val)

    return train_dataset, val_dataset

    
def train_baseline(train_loader, val_loader, config):
    # ---- Train with masking ----
    model = BaselineNNLightning(
        input_dim=config['feature_dim'], 
        hidden_dim=config['nn_hidden_dim'], 
        output_dim=config['label_dim'], 
        lr=config['lr_baseline'], 
        task=config['task'], 
        dropout=config['nn_dropout'],
        mask_random=True, 
        free_indices=config['free_indices']
    )

    ckpt_root = Path(f"./checkpoints/{config['experiment']}/{config['mode']}")
    ckpt_root.mkdir(parents=True, exist_ok=True) 

    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_root), monitor="val_loss", mode="min", save_top_k=1, save_weights_only=False)
    quiet_kwargs = {}
    if not config.get("log_progress", True):
        quiet_kwargs.update({
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        })
    
    trainer = pl.Trainer(
        max_epochs=config['nepochs_baseline'],
        callbacks=[checkpoint],
        accelerator='gpu',
        devices=config['num_gpus'],
        **quiet_kwargs
    )
    trainer.fit(model, train_loader, val_loader)

    # Final model after fine-tuning
    best_finetuned_model = BaselineNNLightning.load_from_checkpoint(checkpoint.best_model_path)
    return best_finetuned_model

# def train_xgboost(X, y, config):
#     x_train = X.cpu().numpy()
#     y_train = y.cpu().numpy()

#     x_vals = x_train[:, :config['feature_dim']]
#     x_mask = x_train[:, config['feature_dim']:]

#     # Set values to nan where mask is 0
#     x_vals[x_mask == 0] = float('nan')
#     y_train = y_train.reshape(-1)  # Flatten labels

#     dtrain = xgb.DMatrix(x_vals, label=y_train)

#     params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#     }

#     # Train first XGBoost model for mean prediction
#     num_round = 100
#     bst = xgb.train(params, dtrain, num_round)

#     return bst

