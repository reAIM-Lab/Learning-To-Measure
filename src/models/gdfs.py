import torch
import torch.nn as nn
from pathlib import Path

import pytorch_lightning as pl
from torch.distributions import Normal, Categorical
from torch.distributions.bernoulli import Bernoulli
from pytorch_lightning.strategies import DDPStrategy

from src.utils.utils import MLP, ConcreteSelector, BaselineNN, generate_uniform_mask

def fit_gdfs(train_loader, val_loader, predictor, config):
    model = GDFSLightning(config, predictor)
    ckpt_root = Path(f"./checkpoints/{config['experiment']}/{config['mode']}/gdfs/")
    ckpt_root.mkdir(parents=True, exist_ok=True) 

    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_root), monitor="val_loss", mode="min", save_top_k=1, save_weights_only=False)

    quiet_kwargs = {}
    if not config.get("log_progress", True):
        quiet_kwargs.update({
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        })
    
    trainer = pl.Trainer(max_epochs=config['nepochs_gdfs'], 
                         callbacks=[checkpoint], 
                         accelerator='gpu', 
                         devices=config['num_gpus'], 
                         log_every_n_steps=10,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         **quiet_kwargs)
    trainer.fit(model, train_loader, val_loader)

    model = GDFSLightning.load_from_checkpoint(checkpoint.best_model_path)
    return model

class GDFSLightning(pl.LightningModule):
    def __init__(self, config, predictor=None):
        super().__init__()
        self.save_hyperparameters(ignore=['predictor'])
        self.x_dim = config['feature_dim']
        self.y_dim = config['label_dim']
        self.task = config['task']
        
        # Set up models and mask layer.
        self.selector = MLP(self.x_dim * 2, self.x_dim, config['nn_hidden_dim'], 2, policy_head=True)

        if predictor is None:
            self.predictor = BaselineNN(self.x_dim * 2, config['nn_hidden_dim'], self.y_dim, task=self.task)
        else:
            self.add_module("predictor", predictor)
        
        # Set up selector layer.
        self.selector_layer = ConcreteSelector()

        self.lr = config['lr_gdfs']
        self.nepochs = config['nepochs_gdfs']
        self.train_predictor = config['train_predictor_gdfs']
        self.temp = config['temp_gdfs']
        self.free_indices = config['free_indices']
        self.num_available = config['num_available_features']

        self.automatic_optimization = False

    def forward(self, x, m):
        x = x * m
        x = torch.cat([x, m], dim=-1)
        return self.selector(x)
    
    def evaluate(self, x, m, y):
        x = x * m
        x = torch.cat([x, m], dim=-1)

        if self.task == 'regression':
            mean, std = self.predictor(x)
            dist = Normal(mean.view(-1), std.view(-1))
            log_probs = dist.log_prob(y)
        elif self.task == 'classification':
            logits = self.predictor(x)
            if self.y_dim == 1:
                probs = torch.sigmoid(logits.view(-1))
                dist = Bernoulli(probs)
                log_probs = dist.log_prob(y)
            else:
                probs = torch.softmax(logits, dim=-1)
                targets = y.argmax(dim=-1)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(targets)

        return dist, log_probs

    def training_step(self, batch, batch_idx):
        x, r, y = batch
        m = torch.zeros_like(x)

        if self.free_indices is not None:
            m[:, self.free_indices] = 1

        opt = self.optimizers()
        opt.zero_grad()

        loss = torch.tensor(0.0, device=x.device)
        for _ in range(self.num_available):
            logits = self(x, m).flatten(1)
            available = r * (1 - m)
            has_available = (available.sum(dim=1) > 0).float()

            logits = logits - 1e8 * (available == 0)
            soft = self.selector_layer(logits, self.temp)
            m_soft = torch.max(m, soft)

            dist, log_prob = self.evaluate(x, m_soft, y)
            log_prob = log_prob * has_available

            if has_available.sum() > 0:
                nll = -(log_prob.sum() / has_available.sum())
                self.manual_backward(nll / self.num_available)
            else:
                nll = torch.tensor(0.0, device=x.device) 

            loss += nll / self.num_available

            m = torch.max(m, self.selector_layer(logits, 1e-6, deterministic=True))

        opt.step()
        self.log("train_loss", loss.detach().float(), prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, r, y = batch
        m = generate_uniform_mask(x.shape[0], self.x_dim, seed=42).to(x.device)
        m = m * r

        if self.free_indices is not None:
            m[:, self.free_indices] = 1

        logits = self(x, m).flatten(1)
        available = r * (1 - m)

        logits = logits - 1e8 * (available == 0)
        m = torch.max(m, self.selector_layer(logits, 1e-6, deterministic=True))
        dist, log_probs = self.evaluate(x, m, y)
        loss = -log_probs.mean()

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.train_predictor:
            opt = torch.optim.Adam(set(list(self.predictor.parameters()) + list(self.selector.parameters())), lr=self.lr)
        else:
            for param in self.predictor.parameters():
                param.requires_grad = False
            opt = torch.optim.Adam(self.selector.parameters(), lr=self.lr)
        return opt

