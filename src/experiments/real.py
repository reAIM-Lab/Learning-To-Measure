import os

from accelerate import Accelerator
from tqdm import tqdm
import torch
from transformers import get_linear_schedule_with_warmup

from src.data.miniboone import load_miniboone_data
from src.data.mimic import load_mimic_data
from src.data.metabric import load_metabric_data
from src.data.mnist import load_mnist_data, MNISTSampler
from src.data.data_utils import RealDataSampler
from src.utils.utils import ConcreteSelector, get_logger
from src.utils.experiment_utils import eval, eval_afa
from src.utils.model_utils import get_model
from src.utils.train_utils import afa_training_step, afa_training_step_random, predictor_training_step
from src.utils.eval_utils import test_model, test_afa, test_afa_bench

def load_data(config):
    if config['experiment'] == 'mimic':
        train_dataset, test_dataset = load_mimic_data(config)
    elif config['experiment'] == 'miniboone':
        train_dataset, test_dataset = load_miniboone_data(config)
    elif config['experiment'] == 'metabric':
        train_dataset, test_dataset = load_metabric_data(config)
    elif config['experiment'] == 'mnist':
        train_dataset, test_dataset = load_mnist_data(config)

    return train_dataset, test_dataset

def get_data_fn(config):
    if config['experiment'] == 'mimic':
        data_fn = load_mimic_data
    elif config['experiment'] == 'miniboone':
        data_fn = load_miniboone_data
    elif config['experiment'] == 'metabric':
        data_fn = load_metabric_data
    elif config['experiment'] == 'mnist':
        data_fn = load_mnist_data

    return data_fn

def real_train(config):
    # load data
    train_dataset, test_dataset = load_data(config)

    config['feature_dim'] = train_dataset.feature_dim
    config['num_available_features'] = train_dataset.num_available_features

    if config['experiment'] == 'mnist':
        sampler = MNISTSampler(train_dataset, test_dataset)
    else:
        sampler = RealDataSampler(train_dataset, test_dataset)
    
    model = get_model(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['num_warmup_steps'], num_training_steps=config['num_steps'])
    accelerator = Accelerator(mixed_precision="bf16")
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    selector_fn = ConcreteSelector()

    best_loss = float('inf')
    train_losses = []
    running_avg_window = config.get('running_avg_window', config['checkpoint_interval'])

    log_dir = config.get('log_dir', '.')
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, f"{config['experiment']}_train.log")
    logger = get_logger(logfilename)
    best_model_path = os.path.join(log_dir, f"best_model_{config['embed']}.pt")

    for step in range(config['num_steps']):
        batch = sampler.sample(batch_size=config['batch_size'], 
                        max_num_points=config['num_points'],
                        num_observed=config['feature_dim'],
                        device=accelerator.device)
        
        loss = predictor_training_step(config, model, sampler, batch, optimizer, scheduler, accelerator, selector_fn)
        train_losses.append(loss)

        if len(train_losses) > running_avg_window:
            train_losses.pop(0)

        if step % config['checkpoint_interval'] == 0:
            avg_train_loss = sum(train_losses) / len(train_losses)
            line = f"[TRAIN] step: {step} | avg_loss: {avg_train_loss:.6f}"
            logger.info(line + '\n')

            if config['verbose']:
                print(line)

            loss, line = eval(config, sampler, model, accelerator)
            logger.info(line + '\n')

            if config['verbose']:
                print(line)
            # Save the model if it is the best so far
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved at step {step} with loss {best_loss:.6f}")
            model.train()

def real_train_afa(config):
    # load data
    train_dataset, test_dataset = load_data(config)

    config['feature_dim'] = train_dataset.feature_dim
    config['num_available_features'] = config['feature_dim']

    # load model
    log_dir = config.get('log_dir', '.')
    best_model_path = os.path.join(log_dir, f"best_model_{config['embed']}.pt")
    #best_model_path = os.path.join(log_dir, f"best_model_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.pt")
    model = get_model(config)

    logfilename = os.path.join(log_dir, f"{config['experiment']}_train_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.log")
    logger = get_logger(logfilename)

    if os.path.exists(best_model_path) and config.get('from_checkpoint', False):
        checkpoint = torch.load(best_model_path, map_location='cuda')
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {best_model_path}" + '\n')
    else:
        logger.info(f"Training from scratch" + '\n')
        config['lr_backbone'] = config['lr_selector'] # If no checkpoint, use the same learning rate for selector and backbone

    if config['experiment'] == 'mnist':
        sampler = MNISTSampler(train_dataset, test_dataset)
    else:
        sampler = RealDataSampler(train_dataset, test_dataset)

    # Only train selector if specified in config
    if config.get('freeze_encoder', False):
        for param in model.parameters():
            param.requires_grad = False

        for param in model.selector.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
        # for param in model.selector.parameters():
        #     param.requires_grad = True
        # for param in model.predictor.parameters():
        #     param.requires_grad = True

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)" + '\n')

    selector_params = list(model.selector.parameters())
    selector_param_ids = set(map(id, selector_params))
    backbone_params = [p for p in model.parameters() if id(p) not in selector_param_ids]

    optimizer = torch.optim.Adam([
        {"params": selector_params, "lr": config['lr_selector']},
        {"params": backbone_params, "lr": config['lr_backbone']}
    ])

    if config['scheduler_afa'] == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['num_steps_afa_warmup'], num_training_steps=config['num_steps_afa'])
    else:
        scheduler = None

    accelerator = Accelerator(mixed_precision="bf16")
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    selector_fn = ConcreteSelector()

    temp_start = config['temp']
    temp_end = 0.1
    if config.get('temp_decay', False):
        # Linear decay from temp_start to temp_end over num_steps_afa
        def get_temp(step):
            progress = step / config['num_steps_afa']
            return temp_start + (temp_end - temp_start) * progress
        temp = temp_start
    else:
        temp = temp_start

    if config['afa_training_strategy'] == 'sequential':
        train_step_fn = afa_training_step
    elif config['afa_training_strategy'] == 'random':
        train_step_fn = afa_training_step_random
    else:
        raise ValueError(f"Invalid AFA training strategy: {config['afa_training_strategy']}")

    best_loss = float('inf')
    train_losses = []
    running_avg_window = config.get('running_avg_window', config['checkpoint_interval_afa'])

    logfilename = os.path.join(log_dir, f"{config['experiment']}_train_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.log")
    logger = get_logger(logfilename)
    best_model_path = os.path.join(log_dir, f"best_model_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.pt")

    for step in range(config['num_steps_afa']):
        batch = sampler.sample(batch_size=config['batch_size_afa'], 
                            max_num_points=config['num_points'], 
                            device=accelerator.device, 
                            num_observed=config['feature_dim'],
                            )

        loss = train_step_fn(config, model, sampler, batch, optimizer, scheduler, accelerator, selector_fn, temp)

        train_losses.append(loss)
        if len(train_losses) > running_avg_window:
            train_losses.pop(0)

        if step % config['checkpoint_interval_afa'] == 0: 
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']

            avg_train_loss = sum(train_losses) / len(train_losses)
            line = f"[TRAIN] step: {step} | avg_loss: {avg_train_loss:.6f} | current_lr: {current_lr:.8f} | current_temp: {temp:.3f}"
            logger.info(line + '\n')

            if config['verbose']:
                print(line)

            loss, line = eval_afa(config, sampler, model, accelerator, temp)
            logger.info(line + '\n')
            
            if config['verbose']:
                print(line)
            #Save the model if it is the best so far
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved at step {step} with loss {best_loss:.6f}")
            
            model.train()
        
        if config['temp_decay']:
            temp = get_temp(step)

def real_test(config):
    # load data
    data_fn = get_data_fn(config)
    
    # Generate test datasets similar to eval, and compute log loss from loaded best model
    if config['experiment'] == 'mnist':
        test_model(config, data_fn, MNISTSampler)
    else:
        test_model(config, data_fn, RealDataSampler)

def real_test_afa(config):
    # load data
    data_fn = get_data_fn(config)
    # Generate test datasets similar to eval, and compute log loss from loaded best model
    if config['experiment'] == 'mnist':
        test_afa(config, data_fn, MNISTSampler)
    else:
        test_afa(config, data_fn, RealDataSampler)

def real_bench_afa(config):
    # load data
    data_fn = get_data_fn(config)
    test_afa_bench(config, data_fn, RealDataSampler)