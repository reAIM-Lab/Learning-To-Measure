import logging
import os

from accelerate import Accelerator
from tqdm import tqdm
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from src.data.data import RBFKernel, GPWithMissingSampler, MaternKernel
from src.utils.utils import ConcreteSelector, get_logger
from src.utils.model_utils import get_model
from src.utils.experiment_utils import eval, eval_afa, get_eval_dir
from src.utils.train_utils import afa_training_step, afa_training_step_random, predictor_training_step
from src.utils.utils import train_baseline, get_loaders

def sim_train(config):
    config['num_available_features'] = config['feature_dim']

    sampler = GPWithMissingSampler(RBFKernel())
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
    logfilename = os.path.join(log_dir, "sim_train.log")
    logger = get_logger(logfilename)
    best_model_path = os.path.join(log_dir, f"best_model_{config['embed']}.pt")

    for step in range(config['num_steps']):
        batch = sampler.sample(batch_size=config['batch_size'], 
                max_num_points=config['num_points'],
                x_dim=config['feature_dim'],
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

def sim_train_afa(config):
    config['num_available_features'] = config['feature_dim']

    log_dir = config.get('log_dir', '.')
    best_model_path = os.path.join(log_dir, f"best_model_{config['embed']}.pt")
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

    sampler = GPWithMissingSampler(RBFKernel())

    # Only train selector if specified in config
    if config.get('freeze_encoder', False):
        for param in model.parameters():
            param.requires_grad = False

        for param in model.selector.parameters():
            param.requires_grad = True
        # for param in model.predictor.parameters():
        #     param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.selector.parameters():
            param.requires_grad = True
        for param in model.predictor.parameters():
            param.requires_grad = True

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
        batch = sampler.sample(batch_size=config['batch_size'], 
                    max_num_points=config['num_points'],
                    x_dim=config['feature_dim'],
                    num_observed=config['feature_dim'],
                    device=accelerator.device,
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
            
            # Save the model if it is the best so far
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved at step {step} with loss {best_loss:.6f}")
            
            model.train()
            if config['temp_decay']:
                temp = get_temp(step)

def sim_test(config):
    # Generate test datasets similar to eval, and compute log loss from loaded best model
    config['free_indices'] = None
    config['num_available_features'] = config['feature_dim']

    # Generate test set path and filename
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-test.tar'
    testset_path = os.path.join(test_path, test_filename)

    # Generate test set if it doesn't exist
    if not os.path.exists(testset_path):
        if config['eval_kernel'] == 'rbf':
            kernel = RBFKernel()
        elif config['eval_kernel'] == 'matern':
            kernel = MaternKernel()
        else:
            raise ValueError(f'Invalid kernel {config["eval_kernel"]}')
        print(f"Generating Test Sets with {config['eval_kernel']} kernel")
        sampler = GPWithMissingSampler(kernel)

        # First generate full batches
        full_batches = []
        for i in tqdm(range(config['test_num_batches']), ascii=True):
            gen = torch.Generator().manual_seed(config['eval_seed']+i)
            num_ctx = torch.randint(50, config['num_points'] - 50, (1,), generator=gen).item()

            full_batches.append(sampler.sample(
                batch_size=1,
                max_num_points=config['num_points'],
                x_dim=config['feature_dim'],
                device='cuda',
                num_observed=config['feature_dim'],
                num_ctx=num_ctx,
                num_tar=config['test_points'],
                seed=config['eval_seed']+i,
                mode='test'))

        # Sequentially acquire features randomly
        test_batches = {i: [] for i in range(1, config['feature_dim'] + 1)}
        for full_batch in full_batches:
            mask = None
            for num_features in range(1, config['feature_dim'] + 1):
                batch, mask = sampler.acquire_features(full_batch, mask, action=None)
                test_batches[num_features].append(batch)

        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        torch.save(test_batches, testset_path)
    else:
        test_batches = torch.load(testset_path, map_location='cuda')
        full_batches = None

    # Load best model
    log_dir = config.get('log_dir', '.')
    best_model_path = os.path.join(log_dir, f"best_model_{config['embed']}.pt")
    model = get_model(config)  # Assumes get_model is defined elsewhere
    checkpoint = torch.load(best_model_path, map_location='cuda')
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    # Compute log loss on test set
    tnpd_metrics = {'nlls': {}, 'mses': {}}
    nn_metrics = {'nlls': {}, 'mses': {}}
    tnpd_predictions = {'means': {}, 'stds': {}, 'gt': {}}
    nn_predictions = {'means': {}, 'stds': {}, 'gt': {}}

    # TNPD results
    with torch.no_grad():
        for num_features, batches in test_batches.items():
            nll_list = []
            mse_list = []
            means_list = []
            stds_list = []
            gt_list = []
            for batch in batches:
                batch = {k: v.cuda() for k, v in batch.items()}
                dist, log_prob = model.predict(batch)
                nll = -log_prob.squeeze().mean().item()
                nll_list.append(nll)
                mse = F.mse_loss(dist.mean, batch['yt']).item()
                mse_list.append(mse)
                means_list.append(dist.mean.squeeze(0).detach().cpu().numpy())
                stds_list.append(dist.stddev.squeeze(0).detach().cpu().numpy())
                gt_list.append(batch['yt'].squeeze().detach().cpu().numpy())
            
            tnpd_metrics['nlls'][num_features] = nll_list
            tnpd_metrics['mses'][num_features] = mse_list
            tnpd_predictions['means'][num_features] = means_list
            tnpd_predictions['stds'][num_features] = stds_list
            tnpd_predictions['gt'][num_features] = gt_list

    # Baseline NN results
    if full_batches is None:
        full_batches = []
        for num_features, batches in test_batches.items():
            if num_features == config['feature_dim']:
                full_batches.extend(batches)

    baseline_dir = os.path.join(log_dir, 'baseline_models')
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    models = []
    for i, batch in enumerate(full_batches):
        baseline_model_path = os.path.join(baseline_dir, f'{config["eval_kernel"]}_baseline_model_simulated_{i}.pt')
        if os.path.exists(baseline_model_path):
            models.append(torch.load(baseline_model_path, weights_only=False))
        else:
            X, R, Y = batch['xc'].cpu().squeeze(0), batch['rc'].cpu().squeeze(0), batch['yc'].cpu().squeeze()
            train_loader, val_loader = get_loaders(X, R, Y, config)
            baseline_model = train_baseline(train_loader, val_loader, config)
            models.append(baseline_model)
            torch.save(baseline_model, baseline_model_path)

    with torch.no_grad():
        for num_features, batches in test_batches.items():
            nll_list = []
            mse_list = []
            means_list = []
            stds_list = []
            gt_list = []
            for i, batch in enumerate(batches):
                baseline_model = models[i]
                baseline_model.to(batch['xt'].device)
                baseline_model.eval()
                X_test = batch['xt'].squeeze(0)
                m_test = batch['maskt'].squeeze(0)
                y_test = batch['yt'].squeeze()
                dist, log_probs = baseline_model.evaluate(X_test, m_test, y_test)
                nll = -log_probs.mean().item()
                nll_list.append(nll)
                mse = F.mse_loss(dist.mean, y_test).item()
                mse_list.append(mse)
                means_list.append(dist.mean.detach().cpu().numpy())
                stds_list.append(dist.stddev.detach().cpu().numpy())
                gt_list.append(y_test.detach().cpu().numpy())
            
            nn_metrics['nlls'][num_features] = nll_list
            nn_metrics['mses'][num_features] = mse_list
            nn_predictions['means'][num_features] = means_list
            nn_predictions['stds'][num_features] = stds_list
            nn_predictions['gt'][num_features] = gt_list

    # def print_metrics(metrics, model_type, num_batches):
    #     print(f"==== [TEST] {model_type} metrics by num_features ====")
    #     for num_features, nll_lists in metrics['nlls'].items():
    #         avg_nll = torch.mean(torch.tensor(nll_lists))
    #         std_err = torch.std(torch.tensor(nll_lists)) / torch.sqrt(torch.tensor(len(nll_lists)))
    #         print(f"[TEST] {model_type} average log loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f} over {num_batches} batches")

    #     for num_features, mse_lists in metrics['mses'].items():
    #         avg_mse = torch.mean(torch.tensor(mse_lists))
    #         std_err = torch.std(torch.tensor(mse_lists)) / torch.sqrt(torch.tensor(len(mse_lists)))
    #         print(f"[TEST] {model_type} average mse (num_features={num_features}): {avg_mse:.6f} ± {std_err:.6f} over {num_batches} batches")

    def print_metrics(metrics, baseline_metrics, model_type, num_batches):
        print(f"==== [TEST] {model_type} metrics by num_features ====")

        abs_impr_dict = {"nlls": {}, "mses": {}}

        # --- NLL ---
        for num_features, nll_lists in metrics['nlls'].items():
            nlls = torch.tensor(nll_lists)
            avg_nll = nlls.mean()
            std_err = nlls.std() / (len(nlls) ** 0.5)

            baseline_nlls = torch.tensor(baseline_metrics['nlls'][num_features])
            impr = baseline_nlls - nlls  # absolute improvement
            avg_impr = impr.mean()
            stderr_impr = impr.std() / (len(impr) ** 0.5)

            print(
                f"[TEST] {model_type} average log loss (num_features={num_features}): "
                f"{avg_nll:.6f} ± {std_err:.6f} over {num_batches} batches | "
                f"absolute improvement vs baseline: {avg_impr:.6f} ± {stderr_impr:.6f}"
            )
            abs_impr_dict["nlls"][num_features] = {
                "abs_impr_avg": avg_impr.item(),
                "abs_impr_stderr": stderr_impr.item()
            }

        # --- MSE ---
        for num_features, mse_lists in metrics['mses'].items():
            mses = torch.tensor(mse_lists)
            avg_mse = mses.mean()
            std_err = mses.std() / (len(mses) ** 0.5)

            baseline_mses = torch.tensor(baseline_metrics['mses'][num_features])
            impr = baseline_mses - mses  # absolute improvement
            avg_impr = impr.mean()
            stderr_impr = impr.std() / (len(impr) ** 0.5)

            print(
                f"[TEST] {model_type} average mse (num_features={num_features}): "
                f"{avg_mse:.6f} ± {std_err:.6f} over {num_batches} batches | "
                f"absolute improvement vs baseline: {avg_impr:.6f} ± {stderr_impr:.6f}"
            )
            abs_impr_dict["mses"][num_features] = {
                "abs_impr_avg": avg_impr.item(),
                "abs_impr_stderr": stderr_impr.item()
            }
        
        return abs_impr_dict
        
    # Print and save all results at the end
    impr = print_metrics(tnpd_metrics, nn_metrics, "ICL", len(test_batches[1]))

    # Optionally, save results to file
    results = {
        "TNPD_nlls": tnpd_metrics['nlls'],
        "NN_nlls": nn_metrics['nlls'],
        "TNPD_mses": tnpd_metrics['mses'],
        "NN_mses": nn_metrics['mses'],
        "TNPD_means": tnpd_predictions['means'],
        "TNPD_stds": tnpd_predictions['stds'],
        "NN_means": nn_predictions['means'],
        "NN_stds": nn_predictions['stds'],
        "TNPD_gt": tnpd_predictions['gt'],
        "NN_gt": nn_predictions['gt'],
        "improvements": impr,
    }
    results_path = os.path.join(log_dir, "test_results.pt")
    torch.save(results, results_path)

def sim_test_afa(config):
    config['free_indices'] = None
    config['num_available_features'] = config['feature_dim']

    # Generate test set path and filename
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-test-afa.tar'
    testset_path = os.path.join(test_path, test_filename)

    if config['eval_kernel'] == 'rbf':
        kernel = RBFKernel()
    elif config['eval_kernel'] == 'matern':
        kernel = MaternKernel()
    else:
        raise ValueError(f'Invalid kernel {config["eval_kernel"]}')

    sampler = GPWithMissingSampler(kernel)

    # Generate test set if it doesn't exist
    if not os.path.exists(testset_path):
        print(f"Generating Test Sets with {config['eval_kernel']} kernel")
        
        test_batches = []
        for i in tqdm(range(config['test_num_batches_afa']), ascii=True):
            gen = torch.Generator().manual_seed(config['eval_seed']+i)
            num_ctx = torch.randint(50, config['num_points'] - 50, (1,), generator=gen).item()

            test_batches.append(sampler.sample(
                batch_size=1,
                max_num_points=config['num_points'],
                x_dim=config['feature_dim'],
                device='cuda',
                num_observed=config['feature_dim'],
                num_ctx=num_ctx,
                seed=config['eval_seed']+i,
                num_tar=config['test_points'],
                mode='test'))
            
        # Save full batches
        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        torch.save(test_batches, testset_path)
    else:
        test_batches = torch.load(testset_path, map_location='cuda')

    # Load best model
    log_dir = config.get('log_dir', '.')
    best_model_path = os.path.join(log_dir, f"best_model_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.pt")
    model = get_model(config)  # Assumes get_model is defined elsewhere
    checkpoint = torch.load(best_model_path, map_location='cuda')
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    # Compare log loss of ICL-based AFA vs random AFA
    tnpd_nlls = {i: [] for i in range(1, config['feature_dim'] + 1)}
    random_nlls = {i: [] for i in range(1, config['feature_dim'] + 1)}
    tnpd_mses = {i: [] for i in range(1, config['feature_dim'] + 1)}
    random_mses = {i: [] for i in range(1, config['feature_dim'] + 1)}

    # ICL-based AFA results
    with torch.no_grad():
        for i, batch in enumerate(test_batches):
            # Move batch tensors to the correct device
            batch = {k: v.to('cuda') for k, v in batch.items()}
            input_batch = sampler.mask_features(batch, num_observed=0)
            mask = input_batch['maskt']

            for num_features in range(1, config['feature_dim'] + 1):    
                outs = model.select_action(input_batch)
                action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats
                input_batch, mask = sampler.acquire_features(batch, mask, action=action)
                dist, log_probs = model.predict(input_batch)
                nll = -log_probs.squeeze().mean().item()
                mse = F.mse_loss(dist.mean.squeeze(), input_batch['yt'].squeeze()).item()
                tnpd_mses[num_features].append(mse)
                tnpd_nlls[num_features].append(nll)

    # Random AFA results
    with torch.no_grad():
        for i, batch in enumerate(test_batches):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            input_batch = sampler.mask_features(batch, num_observed=0)
            mask = input_batch['maskt']

            for num_features in range(1, config['feature_dim'] + 1):
                input_batch, mask = sampler.acquire_features(batch, mask, action=None)
                dist, log_probs = model.predict(input_batch)
                nll = -log_probs.squeeze().mean().item()
                mse = F.mse_loss(dist.mean.squeeze(), input_batch['yt'].squeeze()).item()
                random_mses[num_features].append(mse)
                random_nlls[num_features].append(nll)

    avg_losses = []
    avg_random_losses = []
    # Compute and print average NLL for each number of features
    print("\n==== [TEST] ICL-based AFA avg_log_loss by num_features ====")
    for num_features in range(1, config['feature_dim'] + 1):
        values = tnpd_nlls[num_features]
        avg_nll = sum(values) / len(values)
        avg_losses.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] ICL-based AFA avg_log_loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f}")

    print("\n==== [TEST] Random AFA avg_log_loss by num_features ====")
    for num_features in range(1, config['feature_dim'] + 1):
        values = random_nlls[num_features]
        avg_nll = sum(values) / len(values)
        avg_random_losses.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] Random AFA avg_log_loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f}")

    results = {
        "TNPD": avg_losses,
        "Random": avg_random_losses,
        "TNPD_nlls": tnpd_nlls,
        "Random_nlls": random_nlls,
        "TNPD_mses": tnpd_mses,
        "Random_mses": random_mses
    }
    results_path = os.path.join(log_dir, f"{config['eval_kernel']}_test_results_afa.pt")
    torch.save(results, results_path)

def sim_bench_afa(config):
    config['free_indices'] = None
    config['num_available_features'] = config['feature_dim']

    # Generate test set path and filename
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-bench-afa.tar'
    testset_path = os.path.join(test_path, test_filename)

    if config['eval_kernel'] == 'rbf':
        kernel = RBFKernel()
    elif config['eval_kernel'] == 'matern':
        kernel = MaternKernel()
    else:
        raise ValueError(f'Invalid kernel {config["eval_kernel"]}')

    sampler = GPWithMissingSampler(kernel)

    # Generate test set if it doesn't exist
    if not os.path.exists(testset_path):
        print(f"Generating Test Sets with {config['eval_kernel']} kernel")
        
        test_batches = {}
        test_batches['unseen'] = {num_ctx: [] for num_ctx in config['test_context_lengths']}
        test_batches['missingness'] = {missing_rate: [] for missing_rate in config['test_missingness_rates']}

        full_batches_unseen = []
        for i in tqdm(range(config['test_num_batches_bench_afa']), ascii=True):
            # Sample unseen batch
            full_batches_unseen.append(sampler.sample(
                batch_size=1,
                max_num_points=config['num_points'], 
                x_dim=config['feature_dim'],
                device='cuda',
                num_observed=config['feature_dim'],
                num_ctx=900,
                num_tar=config['test_points'],
                seed=config['test_seed']+i,
                mode='test'
                ))
            
            gen = torch.Generator().manual_seed(config['eval_seed']+i)
            num_ctx = torch.randint(50, config['num_points'] - 50, (1,), generator=gen).item()

            for missing_rate in test_batches['missingness'].keys():
                full_sample = sampler.sample(
                    batch_size=1,
                    max_num_points=config['num_points'],
                    x_dim=config['feature_dim'],
                    device='cuda',
                    num_observed=config['feature_dim'],
                    num_ctx=num_ctx,
                    num_tar=config['test_points'],
                    max_p_missing=missing_rate,
                    seed=config['test_seed']+i,
                    mode='test'
                )
                test_batches['missingness'][missing_rate].append(full_sample)

        for num_ctx in test_batches['unseen'].keys():
            for batch in full_batches_unseen:
                test_batches['unseen'][num_ctx].append(sampler.sample_context(batch, num_ctx))

        # Save full batches
        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        torch.save(test_batches, testset_path)
    else:
        test_batches = torch.load(testset_path, map_location='cuda')

    # Load best model
    log_dir = config.get('log_dir', '.')
    best_model_path = os.path.join(log_dir, f"best_model_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.pt")
    model = get_model(config)  # Assumes get_model is defined elsewhere
    checkpoint = torch.load(best_model_path, map_location='cuda')
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    tnpd_nlls = {num_ctx: [] for num_ctx in test_batches['unseen'].keys()}
    tnpd_nlls_missingness = {missing_rate: [] for missing_rate in test_batches['missingness'].keys()}

    with torch.no_grad():
        # Unseen batches
        for num_ctx in test_batches['unseen'].keys():
            batches = test_batches['unseen'][num_ctx]
            for i, batch in enumerate(batches):
                # Move batch tensors to the correct device
                input_batch = sampler.mask_features(batch, num_observed=0)
                input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
                mask = input_batch['maskt']

                mean_nll = 0.0
                for j in range(1, config['num_available_features'] + 1):    
                    outs = model.select_action(input_batch)
                    action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats
                    input_batch, mask = sampler.acquire_features(batch, mask, action=action)
                    _, log_probs = model.predict(input_batch)
                    nll = -log_probs.squeeze().mean().item()
                    mean_nll += nll
                
                tnpd_nlls[num_ctx].append(mean_nll / config['num_available_features'])

        # Missingness batches
        for missing_rate in test_batches['missingness'].keys():
            batches = test_batches['missingness'][missing_rate]
            for i, batch in enumerate(batches):
                # Move batch tensors to the correct device
                input_batch = sampler.mask_features(batch, num_observed=0)
                input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
                mask = input_batch['maskt']

                mean_nll = 0.0
                for j in range(1, config['num_available_features'] + 1):    
                    outs = model.select_action(input_batch)
                    action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats
                    input_batch, mask = sampler.acquire_features(batch, mask, action=action)
                    _, log_probs = model.predict(input_batch)
                    nll = -log_probs.squeeze().mean().item()
                    mean_nll += nll

                tnpd_nlls_missingness[missing_rate].append(mean_nll / config['num_available_features'])

    avg_losses = []
    avg_losses_missingness = []
    # Compute and print average NLL for each number of features
    print("\n==== [TEST] Unseen ICL-based AFA avg_log_loss by context length ====")
    for num_ctx in tnpd_nlls.keys():
        values = tnpd_nlls[num_ctx]
        avg_nll = sum(values) / len(values)
        avg_losses.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] ICL-based AFA avg_log_loss (num_ctx={num_ctx}): {avg_nll:.6f} ± {std_err:.6f}")

    print("\n==== [TEST] Missingness ICL-based AFA avg_log_loss by missing rate ====")
    for missing_rate in tnpd_nlls_missingness.keys():
        values = tnpd_nlls_missingness[missing_rate]
        avg_nll = sum(values) / len(values)
        avg_losses_missingness.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] ICL-based AFA avg_log_loss (missing_rate={missing_rate}): {avg_nll:.6f} ± {std_err:.6f}")

    results = {
        "Unseen": avg_losses,
        "Unseen_nlls": tnpd_nlls,
        "Missingness": avg_losses_missingness,
        "Missingness_nlls": tnpd_nlls_missingness,
    }
    results_path = os.path.join(log_dir, f"{config['eval_kernel']}_test_results_afa_bench_{config['downstream_task']}.pt")
    torch.save(results, results_path)