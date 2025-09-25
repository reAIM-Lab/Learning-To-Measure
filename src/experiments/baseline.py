import torch
import os
from torch.distributions import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

from src.data.data import RBFKernel, GPWithMissingSampler, MaternKernel
from src.data.miniboone import load_miniboone_data
from src.data.mimic import load_mimic_data
from src.data.metabric import load_metabric_data
from src.data.mnist import load_mnist_data, MNISTSampler
from src.data.data_utils import RealDataSampler

from src.models.gdfs import fit_gdfs
from src.models.dime import fit_dime
from src.utils.utils import get_loaders, train_baseline, get_imputed_datasets
from src.models.dqn import train_dqn
from src.utils.experiment_utils import get_eval_dir

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


def fit_baselines(test_batches, mode, config):
    baseline_models = []
    gdfs_models = []
    dime_models = []

    torch.manual_seed(config['train_seed'])
    
    # Create directory for saved models if it doesn't exist
    baseline_dir = os.path.join(config['log_dir'], 'baseline_models')
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    gdfs_dir = os.path.join(config['log_dir'], 'gdfs_models')
    if not os.path.exists(gdfs_dir):
        os.makedirs(gdfs_dir)

    dime_dir = os.path.join(config['log_dir'], 'dime_models')
    if not os.path.exists(dime_dir):
        os.makedirs(dime_dir)
    
    for i, batch in enumerate(test_batches):

        # Load baseline model
        X, R, Y = batch['xc'].cpu().squeeze(0), batch['rc'].cpu().squeeze(0), batch['yc'].cpu().squeeze()
        train_loader, val_loader = get_loaders(X, R, Y, config)

        if mode == 'real':
            baseline_model_path = os.path.join(baseline_dir, f'baseline_model_afa_{mode}_{config["downstream_task"]}_{i}.pt')
            gdfs_model_path = os.path.join(gdfs_dir, f'gdfs_model_afa_{mode}_{config["downstream_task"]}_{config["train_predictor_gdfs"]}_{i}.pt')
            dime_model_path = os.path.join(dime_dir, f'dime_model_afa_{mode}_{config["downstream_task"]}_{config["train_predictor_gdfs"]}_{i}.pt')
        else:
            if config['experiment'] == 'sim':
                baseline_model_path = os.path.join(baseline_dir, f'{config["eval_kernel"]}_baseline_model_afa_{mode}_{i}.pt')
                gdfs_model_path = os.path.join(gdfs_dir, f'{config["eval_kernel"]}_gdfs_model_afa_{mode}_{config["train_predictor_gdfs"]}_{i}.pt')
                dime_model_path = os.path.join(dime_dir, f'{config["eval_kernel"]}_dime_model_afa_{mode}_{config["train_predictor_gdfs"]}_{i}.pt')
            else:
                baseline_model_path = os.path.join(baseline_dir, f'baseline_model_afa_{mode}_{i}.pt')
                gdfs_model_path = os.path.join(gdfs_dir, f'gdfs_model_afa_{mode}_{config["train_predictor_gdfs"]}_{i}.pt')
                dime_model_path = os.path.join(dime_dir, f'dime_model_afa_{mode}_{config["train_predictor_gdfs"]}_{i}.pt')


        if os.path.exists(baseline_model_path):
            baseline_model = torch.load(baseline_model_path, map_location='cuda', weights_only=False)
        else:
            baseline_model = train_baseline(train_loader, val_loader, config) # Pretrain predictor
            torch.save(baseline_model, baseline_model_path)
        
        if os.path.exists(gdfs_model_path):
            # Load existing model
            gdfs_model = torch.load(gdfs_model_path, weights_only=False)
        else:
            # Fit GDFS model
            gdfs_model = fit_gdfs(train_loader, val_loader, baseline_model.model, config)
            torch.save(gdfs_model, gdfs_model_path)

        if os.path.exists(dime_model_path):
            dime_model = torch.load(dime_model_path, weights_only=False)
        else:
            dime_model = fit_dime(train_loader, val_loader, baseline_model.model, config)
            torch.save(dime_model, dime_model_path)

        baseline_models.append(baseline_model)
        gdfs_models.append(gdfs_model)
        dime_models.append(dime_model)

    return baseline_models, gdfs_models, dime_models

def fit_rl_baselines(test_batches, mode, config):
    dqn_models = []
    torch.manual_seed(config['train_seed'])

    baseline_dir = os.path.join(config['log_dir'], 'baseline_models')
    
    # Create directory for saved models if it doesn't exist
    dqn_dir = os.path.join(config['log_dir'], 'dqn_models')
    if not os.path.exists(dqn_dir):
        os.makedirs(dqn_dir)
    
    for i, batch in enumerate(test_batches):
        X, R, Y = batch['xc'].cpu().squeeze(0), batch['rc'].cpu().squeeze(0), batch['yc'].cpu().squeeze()
        train_dataset, val_dataset = get_imputed_datasets(X, R, Y, config)

        baseline_model_path = os.path.join(baseline_dir, f'baseline_model_afa_{mode}_{config["downstream_task"]}_{i}.pt')
        dqn_model_path = os.path.join(dqn_dir, f'dqn_model_afa_{mode}_{config["downstream_task"]}_{i}.pt')

        baseline_model = torch.load(baseline_model_path, map_location='cuda', weights_only=False)

        if os.path.exists(dqn_model_path):
            dqn_model = torch.load(dqn_model_path, weights_only=False)
        else:
            dqn_model = train_dqn(train_dataset,
                                  val_dataset,
                                  baseline_model.model,
                                  config)
            
            torch.save(dqn_model, dqn_model_path)
        dqn_models.append(dqn_model)

    return dqn_models

def fit_baselines_bench(test_batches, value, config, mode="context_length"):
    baseline_models = []
    gdfs_models = []
    #xgb_models = []
    
    # Create directory for saved models if it doesn't exist
    baseline_dir = os.path.join(config['log_dir'], 'baseline_models')
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    gdfs_dir = os.path.join(config['log_dir'], 'gdfs_models')
    if not os.path.exists(gdfs_dir):
        os.makedirs(gdfs_dir)
    
    for i, batch in enumerate(test_batches):

        # Load baseline model
        X, R, Y = batch['xc'].cpu().squeeze(0), batch['rc'].cpu().squeeze(0), batch['yc'].cpu().squeeze()
        train_loader, val_loader = get_loaders(X, R, Y, config)

        if config['experiment'] == 'sim':
            baseline_model_path = os.path.join(baseline_dir, f'{config["eval_kernel"]}_baseline_model_{mode}_{value}_{i}.pt')
            gdfs_model_path = os.path.join(gdfs_dir, f'{config["eval_kernel"]}_gdfs_model_{mode}_{value}_{i}.pt')
        else:
            if mode == 'missingness_real':
                baseline_model_path = os.path.join(baseline_dir, f'baseline_model_{config["downstream_task"]}_{mode}_{value}_{i}.pt')
                gdfs_model_path = os.path.join(gdfs_dir, f'gdfs_model_{config["downstream_task"]}_{mode}_{value}_{i}.pt')
            else:
                baseline_model_path = os.path.join(baseline_dir, f'baseline_model_{mode}_{value}_{i}.pt')
                gdfs_model_path = os.path.join(gdfs_dir, f'gdfs_model_{mode}_{value}_{i}.pt')

        if os.path.exists(baseline_model_path):
            baseline_model = torch.load(baseline_model_path, map_location='cuda', weights_only=False)
        else:
            baseline_model = train_baseline(train_loader, val_loader, config) # Pretrain predictor
            torch.save(baseline_model, baseline_model_path)
        
        if os.path.exists(gdfs_model_path):
            # Load existing model
            gdfs_model = torch.load(gdfs_model_path, weights_only=False)
        else:
            # Fit GDFS model
            gdfs_model = fit_gdfs(train_loader, val_loader, baseline_model.model, config)
            torch.save(gdfs_model, gdfs_model_path)

        baseline_models.append(baseline_model)
        gdfs_models.append(gdfs_model)

    return baseline_models, gdfs_models

def eval_gdfs(test_batches, max_features, sampler, models, config):
    gdfs_nlls = {i: [] for i in range(1, max_features + 1)}

    if config['task'] == 'regression':
        gdfs_mses = {i: [] for i in range(1, max_features + 1)}
    else:
        gdfs_aurocs = {i: [] for i in range(1, max_features + 1)}

    avg_losses = []
    # Evaluate acquistion performance
    with torch.no_grad():
        for i, batch in enumerate(test_batches):
            # Move batch tensors to the correct device
            batch = {k: v.to('cuda') for k, v in batch.items() if not isinstance(v, list)}
            input_batch = sampler.mask_features(batch, num_observed=0)
            mask = input_batch['maskt']
            gdfs_model = models[i].to('cuda')

            avg_loss = 0.0
            for num_features in range(1, max_features + 1):  
                X_test = input_batch['xt'].squeeze()
                y_test = input_batch['yt'].squeeze()

                outs = gdfs_model(X_test, mask.squeeze(0))
                action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats
                input_batch, mask = sampler.acquire_features(batch, mask, action=action)

                dist, log_probs = gdfs_model.evaluate(input_batch['xt'].squeeze(), mask.squeeze(), y_test)
                nll = -log_probs.squeeze().mean().item()
                gdfs_nlls[num_features].append(nll)

                if config['task'] == 'regression':
                    mse = ((dist.mean - y_test) ** 2).mean().item()
                    gdfs_mses[num_features].append(mse)
                else:
                    auroc = roc_auc_score(y_test.cpu().numpy(), 
                                          dist.probs.cpu().numpy(),
                                          multi_class='ovr',
                                          average='micro')
                    gdfs_aurocs[num_features].append(auroc)

                avg_loss += nll
            avg_losses.append(avg_loss / max_features)

    gdfs_metrics = {
        'nll': gdfs_nlls,
        'mse': gdfs_mses if config['task'] == 'regression' else None,
        'auroc': gdfs_aurocs if config['task'] == 'classification' else None
    }

    return gdfs_metrics, avg_losses

def eval_dime(test_batches, max_features, sampler, models, config):
    dime_nlls = {i: [] for i in range(1, max_features + 1)}

    if config['task'] == 'regression':
        dime_mses = {i: [] for i in range(1, max_features + 1)}
    else:
        dime_aurocs = {i: [] for i in range(1, max_features + 1)}

    with torch.no_grad():
        for i, batch in enumerate(test_batches):
            batch = {k: v.to('cuda') for k, v in batch.items() if not isinstance(v, list)}
            input_batch = sampler.mask_features(batch, num_observed=0)
            mask = input_batch['maskt']
            dime_model = models[i].to('cuda')

            for num_features in range(1, max_features + 1):
                X_test = input_batch['xt'].squeeze()
                y_test = input_batch['yt'].squeeze()

                outs = dime_model(X_test, mask.squeeze(0))
                action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats, pick the largest predicted loss reduction
                input_batch, mask = sampler.acquire_features(batch, mask, action=action)

                dist, log_probs = dime_model.evaluate(input_batch['xt'].squeeze(), mask.squeeze(), y_test)
                nll = -log_probs.squeeze().mean().item()
                dime_nlls[num_features].append(nll)

                if config['task'] == 'regression':
                    mse = ((dist.mean - y_test) ** 2).mean().item()
                    dime_mses[num_features].append(mse)
                else:
                    auroc = roc_auc_score(y_test.cpu().numpy(), 
                                          dist.probs.cpu().numpy(),
                                          multi_class='ovr',
                                          average='micro')
                    dime_aurocs[num_features].append(auroc)

    dime_metrics = {
        'nll': dime_nlls,
        'mse': dime_mses if config['task'] == 'regression' else None,
        'auroc': dime_aurocs if config['task'] == 'classification' else None
    }

    return dime_metrics

def eval_dqn(test_batches, max_features, sampler, models, config):
    dqn_nlls = {i: [] for i in range(1, max_features + 1)}

    if config['task'] == 'regression':
        dqn_mses = {i: [] for i in range(1, max_features + 1)}
    else:
        dqn_aurocs = {i: [] for i in range(1, max_features + 1)}

    avg_losses = []
    # Evaluate acquistion performance
    with torch.no_grad():
        for i, batch in enumerate(test_batches):
            # Move batch tensors to the correct device
            batch = {k: v.to('cuda') for k, v in batch.items() if not isinstance(v, list)}
            input_batch = sampler.mask_features(batch, num_observed=0)
            mask = input_batch['maskt']
            dqn_model = models[i].to('cuda')

            avg_loss = 0.0
            for num_features in range(1, max_features + 1):  
                X_test = input_batch['xt'].squeeze()
                y_test = input_batch['yt'].squeeze()

                q_vals = dqn_model.qnet(X_test, mask.squeeze(0))
                action = torch.argmax(q_vals - 1e6 * mask, dim=-1) # ensure no repeats
                input_batch, mask = sampler.acquire_features(batch, mask, action=action)

                dist, log_probs = dqn_model.evaluate(input_batch['xt'].squeeze(), mask.squeeze(), y_test)
                nll = -log_probs.squeeze().mean().item()
                dqn_nlls[num_features].append(nll)

                if config['task'] == 'regression':
                    mse = ((dist.mean - y_test) ** 2).mean().item()
                    dqn_mses[num_features].append(mse)
                else:
                    auroc = roc_auc_score(y_test.cpu().numpy(), 
                                          dist.probs.cpu().numpy(),
                                          multi_class='ovr',
                                          average='micro')
                    dqn_aurocs[num_features].append(auroc)

                avg_loss += nll
            avg_losses.append(avg_loss / max_features)

    dqn_metrics = {
        'nll': dqn_nlls,
        'mse': dqn_mses if config['task'] == 'regression' else None,
        'auroc': dqn_aurocs if config['task'] == 'classification' else None
    }

    return dqn_metrics


def eval_random(test_batches, max_features, sampler, models, config):
    random_nlls = {i: [] for i in range(1, max_features + 1)}
    random_mses = {i: [] for i in range(1, max_features + 1)}
    random_aurocs = {i: [] for i in range(1, max_features + 1)}

    avg_losses = []
    with torch.no_grad():
        for i, batch in enumerate(test_batches):
            batch = {k: v.to('cuda') for k, v in batch.items() if not isinstance(v, list)}
            input_batch = sampler.mask_features(batch, num_observed=0)
            mask = input_batch['maskt']
            baseline_model = models[i].to('cuda')

            avg_loss = 0.0
            for num_features in range(1, max_features + 1):
                input_batch, mask = sampler.acquire_features(batch, mask, action=None)
                X_test = input_batch['xt'].squeeze(0)
                y_test = input_batch['yt'].squeeze()
                dist, log_probs = baseline_model.evaluate(X_test, mask.squeeze(0), y_test)
                nll = -log_probs.mean().item()
                random_nlls[num_features].append(nll)

                if config['task'] == 'regression':
                    mse = ((dist.mean - y_test) ** 2).mean().item()
                    random_mses[num_features].append(mse)
                else:
                    auroc = roc_auc_score(y_test.cpu().numpy(), 
                                          dist.probs.cpu().numpy(),
                                          multi_class='ovr',
                                          average='micro')
                    random_aurocs[num_features].append(auroc)

                avg_loss += nll
            avg_losses.append(avg_loss / max_features)

    random_metrics = {
        'nll': random_nlls,
        'mse': random_mses if config['task'] == 'regression' else None,
        'auroc': random_aurocs if config['task'] == 'classification' else None
    }

    return random_metrics, avg_losses

def print_results(nlls, max_steps, name, mode="features", task_type="simulated"):    
    if mode == "features":
        print(f"\n==== [TEST] {name} AFA avg_log_loss by num_features ({task_type})====")
    elif mode == "context":
        print(f"\n==== [TEST] {name} AFA avg_log_loss by num_context ({task_type})====")
    elif mode == "missingness":
        print(f"\n==== [TEST] {name} AFA avg_log_loss by missing rate ({task_type})====")
    elif mode == "subgroups":
        print(f"\n==== [TEST] {name} AFA avg_log_loss by subgroups({task_type})====")

    keys = list(nlls.keys())
    for num_steps in range(1, max_steps + 1):
        key = keys[num_steps - 1]
        values = nlls[keys[num_steps - 1]]
        avg_nll = sum(values) / len(values)
        if len(values) > 1:
            std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        else:
            std_err = 0
        if mode == "features":
            print(f"[TEST] {name} AFA avg_log_loss (num_features={key}): {avg_nll:.6f} ± {std_err:.6f}")
        elif mode == "context":
            print(f"[TEST] {name} AFA avg_log_loss (num_context={key}): {avg_nll:.6f} ± {std_err:.6f}")
        elif mode == "missingness":
            print(f"[TEST] {name} AFA avg_log_loss (missing rate={key}): {avg_nll:.6f} ± {std_err:.6f}")
        elif mode == "subgroups":
            print(f"[TEST] {name} AFA avg_log_loss (subgroup={key}): {avg_nll:.6f} ± {std_err:.6f}")

def sim_baseline(config):
    config['free_indices'] = None
    config['num_available_features'] = config['feature_dim']
    # Load data
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

    try:
        test_batches = torch.load(testset_path, map_location='cuda')
        print(f"File found, {len(test_batches)} batches")
    except FileNotFoundError:
        raise FileNotFoundError("Test set not generated")
    
    baseline_models, gdfs_models, dime_models = fit_baselines(test_batches, mode="simulated", config=config)

    gdfs_metrics, _ = eval_gdfs(test_batches, config['feature_dim'], sampler, gdfs_models, config)
    gdfs_nlls = gdfs_metrics['nll']
    gdfs_mses = gdfs_metrics['mse']

    dime_metrics = eval_dime(test_batches, config['feature_dim'], sampler, dime_models, config)
    dime_nlls = dime_metrics['nll']
    dime_mses = dime_metrics['mse']

    random_metrics, _ = eval_random(test_batches, config['feature_dim'], sampler, baseline_models, config)
    random_nlls = random_metrics['nll']
    random_mses = random_metrics['mse']

    print_results(gdfs_nlls, config['feature_dim'], "GDFS")
    print_results(dime_nlls, config['feature_dim'], "DIME")
    print_results(random_nlls, config['feature_dim'], "Random")

    results = {
        "GDFS_nlls": gdfs_nlls,
        "DIME_nlls": dime_nlls,
        "Random_nlls": random_nlls,
        "GDFS_mses": gdfs_mses,
        "DIME_mses": dime_mses,
        "Random_mses": random_mses,
    }
    results_path = os.path.join(config['log_dir'], f"{config['eval_kernel']}_test_results_afa_baselines.pt")
    torch.save(results, results_path)

    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-bench-afa.tar'
    testset_path = os.path.join(test_path, test_filename)

    try:
        test_batches = torch.load(testset_path, map_location='cuda')
    except FileNotFoundError:
        raise FileNotFoundError("Benchmarking set not generated")
    
    gdfs_nlls = {num_ctx: [] for num_ctx in test_batches['unseen'].keys()}
    random_nlls = {num_ctx: [] for num_ctx in test_batches['unseen'].keys()}
    gdfs_nlls_missingness = {missing_rate: [] for missing_rate in test_batches['missingness'].keys()}

    for num_ctx in test_batches['unseen'].keys():
        batches = test_batches['unseen'][num_ctx]
        baseline_models, gdfs_models = fit_baselines_bench(batches, num_ctx, config, mode="context_length")
        
        _, gdfs_avg_losses = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
        gdfs_nlls[num_ctx] = gdfs_avg_losses

        _, random_avg_losses = eval_random(batches, config['num_available_features'], sampler, baseline_models, config)
        random_nlls[num_ctx] = random_avg_losses

    for missing_rate in test_batches['missingness'].keys():
        batches = test_batches['missingness'][missing_rate]
        baseline_models, gdfs_models = fit_baselines_bench(batches, missing_rate, config, mode="missingness")
        _, gdfs_avg_losses = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
        gdfs_nlls_missingness[missing_rate] = gdfs_avg_losses

    print(random_nlls.keys())

    print_results(gdfs_nlls, len(gdfs_nlls.keys()), "GDFS", "context")
    print_results(random_nlls, len(random_nlls.keys()), "Random", "context")
    print_results(gdfs_nlls_missingness, len(gdfs_nlls_missingness.keys()), "GDFS", "missingness")

    results = {
        "GDFS_nlls": gdfs_nlls,
        "Random_nlls": random_nlls,
        "GDFS_nlls_missingness": gdfs_nlls_missingness,
    }
    results_path = os.path.join(config['log_dir'], f"test_results_afa_baselines_bench_{config['downstream_task']}.pt")
    torch.save(results, results_path)

def real_baseline(config):
    # Load data
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-{config["downstream_task"]}-test-afa.tar'
    testset_path = os.path.join(test_path, test_filename)

    train_dataset, test_dataset = load_data(config)

    config['feature_dim'] = train_dataset.feature_dim
    config['num_available_features'] = train_dataset.num_available_features
    config['free_indices'] = train_dataset.free_indices

    if config['experiment'] == 'mnist':
        sampler = MNISTSampler(train_dataset, test_dataset)
    else:
        sampler = RealDataSampler(train_dataset, test_dataset)

    try:
        test_batches = torch.load(testset_path, map_location='cuda')
    except FileNotFoundError:
        raise FileNotFoundError("Test set not generated")
    
    if config['experiment'] != 'mnist':
        batches = test_batches['simulated']
        baseline_models, gdfs_models, dime_models = fit_baselines(batches, mode="simulated", config=config)

        gdfs_metrics, _ = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
        gdfs_nlls = gdfs_metrics['nll']

        dime_metrics = eval_dime(batches, config['num_available_features'], sampler, dime_models, config)
        dime_nlls = dime_metrics['nll']

        random_metrics, _ = eval_random(batches, config['num_available_features'], sampler, baseline_models, config)
        random_nlls = random_metrics['nll']

        print_results(gdfs_nlls, config['num_available_features'], "GDFS", task_type="simulated")
        print_results(dime_nlls, config['num_available_features'], "DIME", task_type="simulated")
        print_results(random_nlls, config['num_available_features'], "Random", task_type="simulated")

        results_sim = {
            "GDFS_nlls": gdfs_nlls,
            "DIME_nlls": dime_nlls,
            "Random_nlls": random_nlls,
        }

        results_path = os.path.join(config['log_dir'], f"test_results_afa_baselines_sim_{config['downstream_task']}.pt")
        torch.save(results_sim, results_path)

    batches = test_batches['real']
    baseline_models, gdfs_models, dime_models = fit_baselines(batches, mode="real", config=config)
    dqn_models = fit_rl_baselines(batches, mode='real', config=config)

    gdfs_metrics, _ = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
    gdfs_nlls_real = gdfs_metrics['nll']
    gdfs_aurocs_real = gdfs_metrics['auroc']

    dime_metrics = eval_dime(batches, config['num_available_features'], sampler, dime_models, config)
    dime_nlls_real = dime_metrics['nll']
    dime_aurocs_real = dime_metrics['auroc']

    dqn_metrics = eval_dqn(batches, config['num_available_features'], sampler, dqn_models, config)
    dqn_nlls_real = dqn_metrics['nll']
    dqn_aurocs_real = dqn_metrics['auroc']

    random_metrics, _ = eval_random(batches, config['num_available_features'], sampler, baseline_models, config)
    random_nlls_real = random_metrics['nll']
    random_aurocs_real = random_metrics['auroc']

    print_results(gdfs_nlls_real, config['num_available_features'], "GDFS", task_type="real")
    print_results(dime_nlls_real, config['num_available_features'], "DIME", task_type="real")
    print_results(dqn_nlls_real, config['num_available_features'], "DQN", task_type="real")
    print_results(random_nlls_real, config['num_available_features'], "Random", task_type="real")

    results_real = {
        "GDFS_nlls_real": gdfs_nlls_real,
        "DIME_nlls_real": dime_nlls_real,
        "DQN_nlls_real": dqn_nlls_real,
        "Random_nlls_real": random_nlls_real,
        "GDFS_aurocs_real": gdfs_aurocs_real,
        "DIME_aurocs_real": dime_aurocs_real,
        "DQN_aurocs_real": dqn_aurocs_real,
        "Random_aurocs_real": random_aurocs_real,
    }

    results_path = os.path.join(config['log_dir'], f"test_results_afa_baselines_{config['downstream_task']}.pt")
    torch.save(results_real, results_path)


def real_baseline_benchmark(config):
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-{config["downstream_task"]}-bench-afa.tar'
    testset_path = os.path.join(test_path, test_filename)

    train_dataset, test_dataset = load_data(config)

    config['feature_dim'] = train_dataset.feature_dim
    config['num_available_features'] = train_dataset.num_available_features
    config['free_indices'] = train_dataset.free_indices

    sampler = RealDataSampler(train_dataset, test_dataset)

    try:
        test_batches = torch.load(testset_path, map_location='cuda')
    except FileNotFoundError:
        raise FileNotFoundError("Benchmarking set not generated")
    
    gdfs_nlls = {num_ctx: [] for num_ctx in test_batches['unseen'].keys()}
    random_nlls = {num_ctx: [] for num_ctx in test_batches['unseen'].keys()}

    gdfs_nlls_missingness = {missing_rate: [] for missing_rate in test_batches['missingness'].keys()}
    gdfs_nlls_missingness_real = {missing_rate: [] for missing_rate in test_batches['missingness_real'].keys()}
    gdfs_nlls_subgroups = {subgroup: [] for subgroup in test_batches['subgroups'].keys()}

    for num_ctx in test_batches['unseen'].keys():
        batches = test_batches['unseen'][num_ctx]
        baseline_models, gdfs_models = fit_baselines_bench(batches, num_ctx, config, mode="context_length")
        
        _, gdfs_avg_losses = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
        gdfs_nlls[num_ctx] = gdfs_avg_losses

        _, random_avg_losses = eval_random(batches, config['num_available_features'], sampler, baseline_models, config)
        random_nlls[num_ctx] = random_avg_losses

    for missing_rate in test_batches['missingness'].keys():
        batches = test_batches['missingness'][missing_rate]
        baseline_models, gdfs_models = fit_baselines_bench(batches, missing_rate, config, mode="missingness")
        _, gdfs_avg_losses = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
        gdfs_nlls_missingness[missing_rate] = gdfs_avg_losses

    for missing_rate in test_batches['missingness_real'].keys():
        batches = test_batches['missingness_real'][missing_rate]
        baseline_models, gdfs_models = fit_baselines_bench(batches, missing_rate, config, mode="missingness_real")
        _, gdfs_avg_losses = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
        gdfs_nlls_missingness_real[missing_rate] = gdfs_avg_losses
        
    for subgroup in test_batches['subgroups'].keys():
        batches = test_batches['subgroups'][subgroup]
        baseline_models, gdfs_models = fit_baselines_bench(batches, subgroup, config, mode="subgroups")
        _, gdfs_avg_losses = eval_gdfs(batches, config['num_available_features'], sampler, gdfs_models, config)
        gdfs_nlls_subgroups[subgroup] = gdfs_avg_losses

    print(random_nlls.keys())

    print_results(gdfs_nlls, len(gdfs_nlls.keys()), "GDFS", "context")
    print_results(random_nlls, len(random_nlls.keys()), "Random", "context")

    print_results(gdfs_nlls_missingness, len(gdfs_nlls_missingness.keys()), "GDFS", "missingness")
    print_results(gdfs_nlls_missingness_real, len(gdfs_nlls_missingness_real.keys()), "GDFS", "missingness_real")
    print_results(gdfs_nlls_subgroups, len(gdfs_nlls_subgroups.keys()), "GDFS", "subgroups")

    results = {
        "GDFS_nlls": gdfs_nlls,
        "Random_nlls": random_nlls,
        "GDFS_nlls_missingness": gdfs_nlls_missingness,
        "GDFS_nlls_missingness_real": gdfs_nlls_missingness_real, 
        "GDFS_nlls_subgroups": gdfs_nlls_subgroups
    }
    results_path = os.path.join(config['log_dir'], f"test_results_afa_baselines_bench_{config['downstream_task']}.pt")
    torch.save(results, results_path)


