import torch
import torch.nn as nn
from pathlib import Path

from torch.distributions import Normal, Categorical
from torch.distributions.bernoulli import Bernoulli
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from src.utils.utils import BaselineNN

def fit_dime(train_loader, val_loader, predictor, config):
    model = DIMELightning(config, predictor)

    ckpt_root = Path(f"./checkpoints/{config['experiment']}/{config['mode']}/dime/")
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

    model = DIMELightning.load_from_checkpoint(checkpoint.best_model_path)
    return model

class DIMELightning(pl.LightningModule):
    def __init__(self, config, predictor=None):
        super().__init__()
        self.save_hyperparameters(ignore=['predictor'])
        self.x_dim = config['feature_dim']
        self.y_dim = config['label_dim']
        self.task = config['task']
        
        # Set up models and mask layer.
        self.value_network = nn.Sequential(
            nn.Linear(self.x_dim * 2, config['nn_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['nn_hidden_dim'], config['nn_hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['nn_hidden_dim'], self.x_dim),
        )

        if predictor is None:
            self.predictor = BaselineNN(self.x_dim * 2, config['nn_hidden_dim'], self.y_dim, task=self.task)
        else:
            self.add_module("predictor", predictor)

        self.lr = config['lr_gdfs']
        self.nepochs = config['nepochs_gdfs']
        self.train_predictor = config['train_predictor_gdfs']
        self.free_indices = config['free_indices']
        self.num_available = config['num_available_features']

        self.automatic_optimization = False

    def forward(self, x, m):
        x = x * m
        x = torch.cat([x, m], dim=-1)
        return self.value_network(x)
    
    
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
    
    def mask_features(self, x, mask):
        x = x * mask
        x = torch.cat([x, mask], dim=-1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, r, y = batch
        m = torch.zeros_like(x)

        if self.free_indices is not None:
            m[:, self.free_indices] = 1

        opt = self.optimizers()
        opt.zero_grad()

        dist, log_probs = self.evaluate(x, m, y)
        nll_prior = -log_probs

        total_loss = 0
        for _ in range(self.num_available):
            pred_cmi = self(x, m).flatten(1)
            available = r * (1 - m)
            has_available = (available.sum(dim=1) > 0).float()

            if has_available.sum() == 0:
                total_loss += torch.tensor(0.0, device=x.device)
                continue

            # pick a random available action
            rand = torch.rand_like(available)
            rand_masked = rand.masked_fill(available == 0, -1)
            actions = rand_masked.argmax(dim=1)
            m = m.clone()
            m.scatter_(1, actions.unsqueeze(1), 1)

            x = x.detach().requires_grad_()
            m = m.detach().requires_grad_()
            dist, log_probs = self.evaluate(x, m, y)
            nll = -log_probs

            delta = nll_prior - nll.detach()
            value_network_loss = nn.functional.mse_loss(pred_cmi[torch.arange(pred_cmi.shape[0]), actions], delta)
            loss = torch.sum(value_network_loss * has_available) / has_available.sum()

            if self.train_predictor:
                loss += torch.sum(nll * has_available) / has_available.sum()

            #loss /= self.x_dim
            self.manual_backward(loss / self.num_available)
            nll_prior = nll.detach()
            total_loss += loss

        opt.step()
        self.log("train_loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, r, y = batch
        m = torch.zeros_like(x)

        if self.free_indices is not None:
            m[:, self.free_indices] = 1

        dist, log_probs = self.evaluate(x, m, y)
        nll_prior = -log_probs

        loss = 0
        for _ in range(self.num_available):
            pred_cmi = self(x, m).flatten(1)
            available = r * (1 - m)
            has_available = (available.sum(dim=1) > 0).float()

            if has_available.sum() == 0:
                continue

            # pick a random available action
            rand = torch.rand_like(available)
            rand_masked = rand.masked_fill(available == 0, -1)
            actions = rand_masked.argmax(dim=1)
            m = m.clone()
            m.scatter_(1, actions.unsqueeze(1), 1)

            dist, log_probs = self.evaluate(x, m, y)
            nll = -log_probs

            delta = nll_prior - nll
            value_network_loss = nn.functional.mse_loss(pred_cmi[torch.arange(pred_cmi.shape[0]), actions], delta)
            loss += torch.sum(value_network_loss * has_available) / has_available.sum()
            loss /= self.num_available

            nll_prior = nll

        self.log("val_loss", loss.detach().float(), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.train_predictor:
            opt = torch.optim.Adam(set(list(self.predictor.parameters()) + list(self.value_network.parameters())), lr=self.lr)
        else:
            for param in self.predictor.parameters():
                param.requires_grad = False
            opt = torch.optim.Adam(self.value_network.parameters(), lr=self.lr)
        return opt

            
    
