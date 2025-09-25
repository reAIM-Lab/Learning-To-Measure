import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from torch.distributions import Bernoulli

from src.utils.utils import get_loaders, train_baseline
from src.utils.model_utils import get_model
from src.utils.experiment_utils import get_eval_dir

def generate_test_batches(config, sampler):
    full_batches = []
    for i in tqdm(range(config['test_num_batches']), ascii=True):
        full_batches.append(sampler.sample_test(
            batch_size=1,
            max_num_points=config['num_points'],
            device='cuda',
            num_observed=config['feature_dim'],
            num_tar=config['test_points'],
            unseen=True,
            seed=config['eval_seed']+i))
        
    full_batches_real = []
    for i in tqdm(range(config['test_num_batches']), ascii=True):
        full_batches_real.append(sampler.sample_test(
            batch_size=1,
            max_num_points=config['num_points'],
            device='cuda',
            num_observed=config['feature_dim'],
            num_tar=config['test_points'],
            unseen=True,
            real_y=True,
            seed=config['eval_seed']+i))
        
    return full_batches, full_batches_real

def compute_kl(p, q, eps = 1e-12):
    if p.ndim == 1: p = p.reshape(-1, 1)
    if q.ndim == 1: q = q.reshape(-1, 1)

    if p.shape[1] == 1:
        p = np.hstack([1.0 - p, p])          # [N,2]
        q = np.hstack([1.0 - q, q])  

    p = np.clip(p, eps, 1); p /= p.sum(1, keepdims=True)
    q = np.clip(q, eps, 1); q /= q.sum(1, keepdims=True)

    kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    kl = kl.mean()
    return kl

def compute_brier(p, q):
    if p.ndim == 1: p = p.reshape(-1, 1)
    if q.ndim == 1: q = q.reshape(-1, 1)

    if p.shape[1] == 1:              # binary, only class-1 probs provided
        diff = (q - p) ** 2          # [N,1]
        return float(np.mean(2.0 * diff))  # equals sum over 2 classes
    else:                             # multiclass
        return float(np.mean(np.sum((q - p)**2, axis=1)))

# def tensor_equal(a: torch.Tensor, b: torch.Tensor, strict=False, atol=1e-7, rtol=1e-7):
#     # move to cpu for consistent comparison
#     a_cpu, b_cpu = a.detach().cpu(), b.detach().cpu()
#     if strict or not a_cpu.is_floating_point():
#         return torch.equal(a_cpu, b_cpu)
#     return torch.allclose(a_cpu, b_cpu, atol=atol, rtol=rtol)

# def compare_runs(batches1, batches2, keys=('x','r','y'), strict=False):
#     assert len(batches1) == len(batches2), "Different number of batches"
#     mismatches = []
#     for i, (b1, b2) in enumerate(zip(batches1, batches2)):
#         for k in keys:
#             if k not in b1 or k not in b2:
#                 mismatches.append((i, k, "missing key"))
#                 continue
#             ok = tensor_equal(b1[k], b2[k], strict=strict)
#             if not ok:
#                 mismatches.append((i, k, "values differ"))
#     return mismatches

def test_model(config, data_fn, sampler_fn):
    # Generate test set path and filename
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-{config["downstream_task"]}-test.tar'
    testset_path = os.path.join(test_path, test_filename)

    train_dataset, test_dataset = data_fn(config)
    config['feature_dim'] = test_dataset.feature_dim
    config['free_indices'] = test_dataset.free_indices
    config['num_available_features'] = test_dataset.num_available_features

    # Generate test set if it doesn't exist
    if not os.path.exists(testset_path):
        print(f"Generating Test Sets for {config['experiment']}")
        sampler = sampler_fn(train_dataset, test_dataset)

        # First generate full batches
        full_batches, full_batches_real = generate_test_batches(config, sampler)
        
        #full_batches_2, full_batches_real_2 = generate_test_batches(config, sampler)
        # mism = compare_runs(full_batches, full_batches_2, keys=('x','r','y'), strict=False)

        # if not mism:
        #     print("no mismatch")
        # else:
        #     print(f"Found {len(mism)} mismatches:")
        #     for i, k, msg in mism[:10]:
        #         print(f"  batch {i}, key '{k}': {msg}")

        # mism = compare_runs(full_batches_real, full_batches_real_2, keys=('x','r','y'), strict=False)

        # if not mism:
        #     print("no mismatch")
        # else:
        #     print(f"Found {len(mism)} mismatches:")
        #     for i, k, msg in mism[:10]:
        #         print(f"  batch {i}, key '{k}': {msg}")

        # Sequentially acquire features randomly
        test_batches = {}
        test_batches['simulated'] = {i: [] for i in range(1, config['num_available_features'] + 1)}
        test_batches['real'] = {i: [] for i in range(1, config['num_available_features'] + 1)}

        for i, sample in enumerate(full_batches):
            mask = None
            for num_features in range(1, config['num_available_features'] + 1):
                batch, mask = sampler.acquire_features(sample, mask, action=None, seed=config['eval_seed']+i)
                test_batches['simulated'][num_features].append(batch)

        for sample in full_batches_real:
            mask = None
            for num_features in range(1, config['num_available_features'] + 1):
                batch, mask = sampler.acquire_features(sample, mask, action=None, seed=config['eval_seed']+i)
                test_batches['real'][num_features].append(batch)

        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        torch.save(test_batches, testset_path)
    else:
        test_batches = torch.load(testset_path, map_location='cuda')
        full_batches = None
        full_batches_real = None

    # Compute log loss on test set
    tnpd_metrics_simulated = {'nlls': {}, 'auroc': {}, 'kl': {}, 'brier': {}}
    tnpd_metrics_real = {'nlls': {}, 'auroc': {}, 'brier': {}}
    nn_metrics_simulated = {'nlls': {}, 'auroc': {}, 'kl': {}, 'brier': {}}
    nn_metrics_real = {'nlls': {}, 'auroc': {}, 'brier': {}}

    # Load best model
    log_dir = config.get('log_dir', '.')
    #best_model_path = os.path.join(log_dir, f"best_model_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.pt")
    best_model_path = os.path.join(log_dir, f"best_model_set.pt")
    model = get_model(config)  # Assumes get_model is defined elsewhere
    checkpoint = torch.load(best_model_path, map_location='cuda')
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    # TNPD results
    with torch.no_grad():
        for num_features, batches in test_batches['simulated'].items():
            nll_list = []
            auroc_list = []
            kl_list = []
            brier_list = []
            for batch in batches:
                batch = {k: v.cuda() for k, v in batch.items()}
                dist, log_probs = model.predict(batch)
                nll = -log_probs.squeeze().mean().item()
                nll_list.append(nll)

                auroc = roc_auc_score(batch['yt'].squeeze().cpu().numpy(), 
                                      dist.probs.squeeze().cpu().numpy(),
                                      multi_class='ovr',
                                      average='micro')
                auroc_list.append(auroc)

                p = batch['pt'].squeeze().cpu().numpy()   # [N,K] or [N,1]
                q = dist.probs.squeeze().cpu().numpy()
                kl = compute_kl(p, q)
                kl_list.append(kl)
                brier = compute_brier(p, q)
                brier_list.append(brier)

            tnpd_metrics_simulated['nlls'][num_features] = nll_list
            tnpd_metrics_simulated['auroc'][num_features] = auroc_list
            tnpd_metrics_simulated['kl'][num_features] = kl_list
            tnpd_metrics_simulated['brier'][num_features] = brier_list

        for num_features, batches in test_batches['real'].items():
            nll_list = []
            auroc_list = []
            brier_list = []
            for batch in batches:
                batch = {k: v.cuda() for k, v in batch.items()}
                dist, log_probs = model.predict(batch)
                nll = -log_probs.squeeze().mean().item()
                nll_list.append(nll)
                auroc = roc_auc_score(batch['yt'].squeeze().cpu().numpy(), dist.probs.squeeze().cpu().numpy())
                auroc_list.append(auroc)
                
                p = batch['yt'].squeeze().cpu().numpy()   # [N,K] or [N,1]
                q = dist.probs.squeeze().cpu().numpy()
                brier = compute_brier(p, q)
                brier_list.append(brier)
                
            tnpd_metrics_real['nlls'][num_features] = nll_list
            tnpd_metrics_real['auroc'][num_features] = auroc_list
            tnpd_metrics_real['brier'][num_features] = brier_list

    # Baseline NN results
    if full_batches is None:
        full_batches = []
        for num_features, batches in test_batches['simulated'].items():
            if num_features == config['num_available_features']:
                full_batches.extend(batches)
    
    if full_batches_real is None:
        full_batches_real = []
        for num_features, batches in test_batches['real'].items():
            if num_features == config['num_available_features']:
                full_batches_real.extend(batches)

    baseline_dir = os.path.join(log_dir, 'baseline_models')
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    models = []
    for i, batch in enumerate(full_batches):
        baseline_model_path = os.path.join(baseline_dir, f'baseline_model_simulated_{config["downstream_task"]}_{i}.pt')
        if os.path.exists(baseline_model_path):
            models.append(torch.load(baseline_model_path, weights_only=False))
        else:
            X, R, Y = batch['xc'].cpu().squeeze(0), batch['rc'].cpu().squeeze(0), batch['yc'].cpu().squeeze()
            print(X.shape, Y.shape)
            train_loader, val_loader = get_loaders(X, R, Y, config)
            baseline_model = train_baseline(train_loader, val_loader, config)
            models.append(baseline_model)
            torch.save(baseline_model, baseline_model_path)

    models_real = []
    for i, batch in enumerate(full_batches_real):
        baseline_model_path = os.path.join(baseline_dir, f'baseline_model_real_{config["downstream_task"]}_{i}.pt')
        if os.path.exists(baseline_model_path):
            models_real.append(torch.load(baseline_model_path, weights_only=False))
        else:
            X, R, Y = batch['xc'].cpu().squeeze(0), batch['rc'].cpu().squeeze(0), batch['yc'].cpu().squeeze()
            print(X.shape, Y.shape)
            train_loader, val_loader = get_loaders(X, R, Y, config)
            baseline_model = train_baseline(train_loader, val_loader, config)
            models_real.append(baseline_model)
            torch.save(baseline_model, baseline_model_path)

    for num_features, batches in test_batches['simulated'].items():
        nll_list = []
        auroc_list = []
        kl_list = []
        brier_list = []
        for i, batch in enumerate(batches):
            baseline_model = models[i]
            baseline_model.to(batch['xt'].device)
            baseline_model.eval()
            with torch.no_grad():
                X_test = batch['xt'].squeeze(0)
                m_test = batch['maskt'].squeeze(0)
                y_test = batch['yt'].squeeze()

                dist, log_probs = baseline_model.evaluate(X_test, m_test, y_test)
                nll = -log_probs.mean().item()
            nll_list.append(nll)
            auroc = roc_auc_score(y_test.cpu().numpy(), 
                                  dist.probs.squeeze().cpu().numpy(),
                                  multi_class='ovr',
                                  average='micro')
            auroc_list.append(auroc)

            p = batch['pt'].squeeze().cpu().numpy()   # [N,K] or [N,1]
            q = dist.probs.squeeze().cpu().numpy()
            kl = compute_kl(p, q)
            kl_list.append(kl)
            brier = compute_brier(p, q)
            brier_list.append(brier)

        nn_metrics_simulated['nlls'][num_features] = nll_list
        nn_metrics_simulated['auroc'][num_features] = auroc_list
        nn_metrics_simulated['kl'][num_features] = kl_list
        nn_metrics_simulated['brier'][num_features] = brier_list

    for num_features, batches in test_batches['real'].items():
        nll_list = []
        auroc_list = []
        brier_list = []
        for i, batch in enumerate(batches):
            baseline_model = models_real[i]
            baseline_model.to(batch['xt'].device)
            baseline_model.eval()
            with torch.no_grad():
                X_test = batch['xt'].squeeze(0)
                m_test = batch['maskt'].squeeze(0)
                y_test = batch['yt'].squeeze()

                dist, log_probs = baseline_model.evaluate(X_test, m_test, y_test)
                nll = -log_probs.mean().item()
            nll_list.append(nll)
            auroc = roc_auc_score(y_test.cpu().numpy(), dist.probs.squeeze().cpu().numpy())
            auroc_list.append(auroc)

            p = y_test.cpu().numpy()   # [N,K] or [N,1]
            q = dist.probs.squeeze().cpu().numpy()
            brier = compute_brier(p, q)
            brier_list.append(brier)

        nn_metrics_real['nlls'][num_features] = nll_list
        nn_metrics_real['auroc'][num_features] = auroc_list
        nn_metrics_real['brier'][num_features] = brier_list

    def print_metrics(metrics, baseline_metrics, model_type, num_batches):
        print(f"==== [TEST] {model_type} metrics by num_features ====")

        impr_dict = {"nlls": {}, "aurocs": {}, "kl": {}, "briers":{}}

        # --- NLL ---
        for num_features, nll_lists in metrics['nlls'].items():
            nlls = torch.tensor(nll_lists)
            avg_nll = nlls.mean()
            std_err = nlls.std() / (len(nlls) ** 0.5)

            baseline_nlls = torch.tensor(baseline_metrics['nlls'][num_features])
            abs_impr = baseline_nlls - nlls
            avg_impr = abs_impr.mean()
            stderr_impr = abs_impr.std() / (len(abs_impr) ** 0.5)

            print(
                f"[TEST] {model_type} average log loss (num_features={num_features}): "
                f"{avg_nll:.6f} ± {std_err:.6f} over {num_batches} batches | "
                f"absolute improvement vs baseline: {avg_impr:.6f} ± {stderr_impr:.6f}"
            )
            impr_dict["nlls"][num_features] = {
                "abs_impr_avg": avg_impr.item(),
                "abs_impr_stderr": stderr_impr.item()
            }

        # --- AUROC ---
        for num_features, auroc_lists in metrics['auroc'].items():
            aurocs = torch.tensor(auroc_lists)
            avg_auroc = aurocs.mean()
            std_err = aurocs.std() / (len(aurocs) ** 0.5)

            baseline_aurocs = torch.tensor(baseline_metrics['auroc'][num_features])
            abs_impr = aurocs - baseline_aurocs
            avg_impr = abs_impr.mean()
            stderr_impr = abs_impr.std() / (len(abs_impr) ** 0.5)

            print(
                f"[TEST] {model_type} average auroc (num_features={num_features}): "
                f"{avg_auroc:.6f} ± {std_err:.6f} over {num_batches} batches | "
                f"absolute improvement vs baseline: {avg_impr:.6f} ± {stderr_impr:.6f}"
            )

            impr_dict["aurocs"][num_features] = {
                "abs_impr_avg": avg_impr.item(),
                "abs_impr_stderr": stderr_impr.item()
            }

        # Brier
        for num_features, brier_lists in metrics['brier'].items():
            briers = torch.tensor(brier_lists)
            avg_brier = briers.mean()
            std_err = briers.std() / (len(briers) ** 0.5)

            baseline_briers = torch.tensor(baseline_metrics['brier'][num_features])
            abs_impr = baseline_briers - briers
            avg_impr = abs_impr.mean()
            stderr_impr = abs_impr.std() / (len(abs_impr) ** 0.5)

            print(
                f"[TEST] {model_type} average brier (num_features={num_features}): "
                f"{avg_brier:.6f} ± {std_err:.6f} over {num_batches} batches | "
                f"absolute improvement vs baseline: {avg_impr:.6f} ± {stderr_impr:.6f}"
            )

            impr_dict["briers"][num_features] = {
                "abs_impr_avg": avg_impr.item(),
                "abs_impr_stderr": stderr_impr.item()
            }

        if "kl" in metrics:
            for num_features, kl_lists in metrics['kl'].items():
                kls = torch.tensor(kl_lists)
                avg_kl = kls.mean()
                std_err = kls.std() / (len(kls) ** 0.5)

                baseline_kls = torch.tensor(baseline_metrics['kl'][num_features])
                abs_impr = baseline_kls - kls
                avg_impr = abs_impr.mean()
                stderr_impr = abs_impr.std() / (len(abs_impr) ** 0.5)

                print(
                    f"[TEST] {model_type} average KL (num_features={num_features}): "
                    f"{avg_kl:.6f} ± {std_err:.6f} over {num_batches} batches | "
                    f"absolute improvement vs baseline: {avg_impr:.6f} ± {stderr_impr:.6f}"
                )
                impr_dict["kl"][num_features] = {
                    "abs_impr_avg": avg_impr.item(),
                    "abs_impr_stderr": stderr_impr.item()
                }
        
        return impr_dict
    
    impr_simulated = print_metrics(tnpd_metrics_simulated, nn_metrics_simulated, "ICL", len(test_batches['simulated'][num_features]))
    impr_real = print_metrics(tnpd_metrics_real, nn_metrics_real, "ICL", len(test_batches['real'][num_features]))

    # Optionally, save results to file
    results_sim = {
        "TNPD_metrics_simulated": tnpd_metrics_simulated,
        "NN_metrics_simulated": nn_metrics_simulated,
        "improvements_simulated": impr_simulated,
    }
    results_real = {
        "TNPD_metrics_real": tnpd_metrics_real,
        "NN_metrics_real": nn_metrics_real,
        "improvements_real": impr_real
    }
    results_path = os.path.join(log_dir, f"test_results_sim_{config['downstream_task']}.pt")
    torch.save(results_sim, results_path)

    results_path = os.path.join(log_dir, f"test_results_{config['downstream_task']}.pt")
    torch.save(results_real, results_path)

def test_afa(config, data_fn, sampler_fn):
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-{config["downstream_task"]}-test-afa.tar'
    testset_path = os.path.join(test_path, test_filename)

    train_dataset, test_dataset = data_fn(config)
    config['feature_dim'] = test_dataset.feature_dim
    config['num_available_features'] = test_dataset.num_available_features

    sampler = sampler_fn(train_dataset, test_dataset)

    # Generate test set if it doesn't exist
    if not os.path.exists(testset_path):
        print(f"Generating Test Sets for {config['experiment']}")
        
        test_batches = {}
        test_batches['simulated'] = []
        test_batches['real'] = []

        for i in tqdm(range(config['test_num_batches_afa']), ascii=True):
            test_batches['simulated'].append(sampler.sample_test(
                batch_size=1,
                max_num_points=config['num_points'],
                device='cuda',
                num_observed=config['feature_dim'],
                min_informative=config['test_min_informative'],
                num_tar=config['test_points'],
                unseen=True,
                seed=config['test_seed']+i))

            test_batches['real'].append(sampler.sample_test(
                batch_size=1,
                max_num_points=config['num_points'],
                device='cuda',
                num_observed=config['feature_dim'],
                num_tar=config['test_points'],
                real_y=True,
                unseen=True,
                seed=config['test_seed']+i))
            
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

    # Print number of model parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")

    # Compare log loss of ICL-based AFA vs random AFA
    tnpd_nlls = {i: [] for i in range(1, config['num_available_features'] + 1)}
    tnpd_static_nlls_real = {i: [] for i in range(1, config['num_available_features'] + 1)}
    tnpd_nlls_real = {i: [] for i in range(1, config['num_available_features'] + 1)}
    random_nlls = {i: [] for i in range(1, config['num_available_features'] + 1)}
    random_nlls_real = {i: [] for i in range(1, config['num_available_features'] + 1)}

    tnpd_aurocs = {i: [] for i in range(1, config['num_available_features'] + 1)}
    tnpd_static_auroc_real = {i: [] for i in range(1, config['num_available_features'] + 1)}
    tnpd_aurocs_real = {i: [] for i in range(1, config['num_available_features'] + 1)}
    random_aurocs = {i: [] for i in range(1, config['num_available_features'] + 1)}
    random_aurocs_real = {i: [] for i in range(1, config['num_available_features'] + 1)}

    # ICL-based AFA results
    batches = test_batches['simulated']
    for i, batch in enumerate(batches):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        input_batch = sampler.mask_features(batch, num_observed=0)
        mask = input_batch['maskt']

        with torch.no_grad():
            for num_features in range(1, config['num_available_features'] + 1):    
                outs = model.select_action(input_batch)
                action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats
                input_batch, mask = sampler.acquire_features(batch, mask, action=action)
                dist, log_probs = model.predict(input_batch)
                nll = -log_probs.squeeze().mean().item()
                auroc = roc_auc_score(batch['yt'].squeeze().cpu().numpy(), 
                                      dist.probs.squeeze().cpu().numpy(),
                                      multi_class='ovr',
                                      average='micro')
                tnpd_aurocs[num_features].append(auroc)
                tnpd_nlls[num_features].append(nll)

    # Random AFA results
    batches = test_batches['simulated']
    with torch.no_grad():
        for i, batch in enumerate(batches):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            input_batch = sampler.mask_features(batch, num_observed=0)
            mask = input_batch['maskt']

            for num_features in range(1, config['num_available_features'] + 1):
                input_batch, mask = sampler.acquire_features(batch, mask, action=None, seed=config['test_seed']+i)
                dist, log_probs = model.predict(input_batch)
                nll = -log_probs.squeeze().mean().item()
                auroc = roc_auc_score(batch['yt'].squeeze().cpu().numpy(), 
                                      dist.probs.squeeze().cpu().numpy(),
                                      multi_class='ovr',
                                      average='micro')
                random_aurocs[num_features].append(auroc)
                random_nlls[num_features].append(nll)

    batches = test_batches['real']
    for i, batch in enumerate(batches):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        input_batch = sampler.mask_features(batch, num_observed=0)
        mask = input_batch['maskt']

        with torch.no_grad():
            for num_features in range(1, config['num_available_features'] + 1):    
                outs = model.select_action(input_batch)
                action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats
                input_batch, mask = sampler.acquire_features(batch, mask, action=action)
                dist, log_probs = model.predict(input_batch)
                nll = -log_probs.squeeze().mean().item()
                auroc = roc_auc_score(batch['yt'].squeeze().cpu().numpy(), 
                                      dist.probs.squeeze().cpu().numpy(),
                                      multi_class='ovr',
                                      average='micro')
                tnpd_aurocs_real[num_features].append(auroc)
                tnpd_nlls_real[num_features].append(nll)

    batches = test_batches['real']
    for i, batch in enumerate(batches):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        input_batch = sampler.mask_features(batch, num_observed=0)
        mask = input_batch['maskt']                  # shape [..., K]

        with torch.no_grad():
            for num_features in range(1, config['num_available_features'] + 1):
                outs = model.select_action(input_batch)
                action = torch.argmax(outs - 1e6 * mask, dim=-1)

                picked = action.reshape(-1)            # flatten
                K = action.size(-1)
                counts = torch.bincount(picked, minlength=K).cpu()
                a_star = int(counts.argmax().item())

                # Apply the same action to every item this step
                action = action.new_full(action.shape, a_star)  # same shape/dtype as per_sample_action
                input_batch, mask = sampler.acquire_features(batch, mask, action=action)

                dist, log_probs = model.predict(input_batch)
                nll = -log_probs.squeeze().mean().item()
                auroc = roc_auc_score(batch['yt'].squeeze().cpu().numpy(), 
                                      dist.probs.squeeze().cpu().numpy(),
                                      multi_class='ovr',
                                      average='micro')
                tnpd_static_auroc_real[num_features].append(auroc)
                tnpd_static_nlls_real[num_features].append(nll)

    for i, batch in enumerate(batches):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        input_batch = sampler.mask_features(batch, num_observed=0)
        mask = input_batch['maskt']

        with torch.no_grad():
            for num_features in range(1, config['num_available_features'] + 1):    
                input_batch, mask = sampler.acquire_features(batch, mask, action=None, seed=config['test_seed']+i)
                dist, log_probs = model.predict(input_batch)
                nll = -log_probs.squeeze().mean().item()
                auroc = roc_auc_score(batch['yt'].squeeze().cpu().numpy(), 
                                      dist.probs.squeeze().cpu().numpy(),
                                      multi_class='ovr',
                                      average='micro')
                random_aurocs_real[num_features].append(auroc)
                random_nlls_real[num_features].append(nll)

    avg_losses = []
    avg_random_losses = []
    avg_losses_real = []
    avg_random_losses_real = []
    # Compute and print average NLL for each number of features
    print("\n==== [TEST] ICL-based AFA avg_log_loss by num_features ====")
    for num_features in range(1, config['num_available_features'] + 1):
        values = tnpd_nlls[num_features]
        avg_nll = sum(values) / len(values)
        avg_losses.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] ICL-based AFA avg_log_loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f}")

    print("\n==== [TEST] Random AFA avg_log_loss by num_features ====")
    for num_features in range(1, config['num_available_features'] + 1):
        values = random_nlls[num_features]
        avg_nll = sum(values) / len(values)
        avg_random_losses.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] Random AFA avg_log_loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f}")

    print("\n==== [TEST] ICL-based AFA avg_log_loss by num_features (real) ====")
    for num_features in range(1, config['num_available_features'] + 1):
        values = tnpd_nlls_real[num_features]
        avg_nll = sum(values) / len(values)
        avg_losses_real.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] ICL-based AFA avg_log_loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f}")

    print("\n==== [TEST] Static ICL-based AFA avg_log_loss by num_features (real) ====")
    for num_features in range(1, config['num_available_features'] + 1):
        values = tnpd_static_nlls_real[num_features]
        avg_nll = sum(values) / len(values)
        #avg_losses.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] Static ICL-based AFA avg_log_loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f}")

    print("\n==== [TEST] ICL-based AFA avg_log_loss by num_features (real) ====")
    for num_features in range(1, config['num_available_features'] + 1):
        values = random_nlls_real[num_features]
        avg_nll = sum(values) / len(values)
        avg_random_losses_real.append(avg_nll)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] Random AFA avg_log_loss (num_features={num_features}): {avg_nll:.6f} ± {std_err:.6f}")

    results_sim = {
        "TNPD": avg_losses,
        "Random": avg_random_losses,
        "TNPD_nlls": tnpd_nlls,
        "Random_nlls": random_nlls,
        "TNPD_aurocs": tnpd_aurocs,
        "Random_aurocs": random_aurocs
    }
    results_real = {
        "TNPD_nlls_real": tnpd_nlls_real,
        "TNPD_avg_losses_real": avg_losses_real,
        "Random_nlls_real": random_nlls_real,
        "TNPD_aurocs_real": tnpd_aurocs_real,
        "Random_aurocs_real": random_aurocs_real,
        "TNPD_static_aurocs": tnpd_static_auroc_real,
        "TNPD_static_nlls": tnpd_static_nlls_real,
    }
    results_path = os.path.join(log_dir, f"test_results_afa_sim_{config['downstream_task']}.pt")
    torch.save(results_sim, results_path)

    results_path = os.path.join(log_dir, f"test_results_afa_{config['downstream_task']}.pt")
    torch.save(results_real, results_path)

def test_afa_bench(config, data_fn, sampler_fn):
    test_path = get_eval_dir(config)
    test_filename = f'{config["eval_kernel"]}-seed{config["test_seed"]}-{config["downstream_task"]}-bench-afa.tar'
    testset_path = os.path.join(test_path, test_filename)

    train_dataset, test_dataset = data_fn(config)
    config['feature_dim'] = test_dataset.feature_dim
    config['num_available_features'] = test_dataset.num_available_features

    sampler = sampler_fn(train_dataset, test_dataset)

        # Generate test set if it doesn't exist
    if not os.path.exists(testset_path):
        print(f"Generating Test Sets for {config['experiment']}")
        
        test_batches = {}
        test_batches['unseen'] = {num_ctx: [] for num_ctx in config['test_context_lengths']}
        test_batches['missingness'] = {missing_rate: [] for missing_rate in config['test_missingness_rates']}
        test_batches['missingness_real'] = {missing_rate: [] for missing_rate in config['test_missingness_rates']}
        test_batches['subgroups'] = {subgroup: [] for subgroup in config['test_subgroups']}
        
        full_batches_unseen = []
        for i in tqdm(range(config['test_num_batches_bench_afa']), ascii=True):
            # Sample unseen batch
            full_batches_unseen.append(sampler.sample_test(
                batch_size=1,
                max_num_points=config['num_points'], 
                device='cuda',
                num_observed=config['feature_dim'],
                num_ctx=900,
                num_tar=config['test_points'],
                real_y=True,
                unseen=True,
                seed=config['test_seed']+i,
                ))
            
            g = torch.Generator(device="cpu")
            g.manual_seed(config['test_seed']+i)
            num_informative = torch.randint(
                low=1,
                high=config['num_available_features']-1,
                size=(1,),
                generator=g
            ).item()

            for subgroup in test_batches['subgroups'].keys():
                full_sample = sampler.sample_test(
                    batch_size=1,
                    max_num_points=config['num_points'],
                    device='cuda',
                    num_observed=config['feature_dim'],
                    num_tar=config['test_points'],
                    subgroups=subgroup,
                    unseen=True,
                    max_p_missing=0.2,
                    min_informative=num_informative,
                    max_informative=num_informative,
                    return_groups=True,
                    seed=config['test_seed']+i,
                )
                test_batches['subgroups'][subgroup].append(full_sample)

            for missing_rate in test_batches['missingness'].keys():
                full_sample = sampler.sample_test(
                    batch_size=1,
                    max_num_points=config['num_points'],
                    device='cuda',
                    num_observed=config['feature_dim'],
                    num_tar=config['test_points'],
                    unseen=True,
                    max_p_missing=missing_rate,
                    seed=config['test_seed']+i,
                )
                test_batches['missingness'][missing_rate].append(full_sample)

        for num_ctx in test_batches['unseen'].keys():
            for batch in full_batches_unseen:
                test_batches['unseen'][num_ctx].append(sampler.sample_context(batch, num_ctx))

        for i in tqdm(range(config['test_num_batches_bench_real_afa']), ascii=True):
            for missing_rate in test_batches['missingness_real'].keys():
                full_sample = sampler.sample_test(
                    batch_size=1,
                    max_num_points=config['num_points'],
                    device='cuda',
                    num_observed=config['feature_dim'],
                    num_tar=config['test_points'],
                    unseen=True,
                    real_y=True,
                    max_p_missing=missing_rate,
                    seed=config['test_seed']+i,
                )
                test_batches['missingness_real'][missing_rate].append(full_sample)

        # Save full batches
        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        torch.save(test_batches, testset_path)
    else:
        test_batches = torch.load(testset_path, map_location='cuda')

    log_dir = config.get('log_dir', '.')
    best_model_path = os.path.join(log_dir, f"best_model_afa_{config['afa_training_strategy']}_{config['freeze_encoder']}.pt")
    model = get_model(config)  # Assumes get_model is defined elsewhere
    checkpoint = torch.load(best_model_path, map_location='cuda')
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    tnpd_nlls = {num_ctx: [] for num_ctx in test_batches['unseen'].keys()}
    tnpd_nlls_missingness = {missing_rate: [] for missing_rate in test_batches['missingness'].keys()}
    tnpd_nlls_missingness_real = {missing_rate: [] for missing_rate in test_batches['missingness_real'].keys()}
    tnpd_nlls_subgroups = {subgroup: [] for subgroup in test_batches['subgroups'].keys()}

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

                    dist, log_probs = model.predict(input_batch)
                    nll = -log_probs.sum(-1).mean().item()
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
                    dist, log_probs = model.predict(input_batch)
                    nll = -log_probs.sum(-1).mean().item()
                    mean_nll += nll

                tnpd_nlls_missingness[missing_rate].append(mean_nll / config['num_available_features'])

        for missing_rate in test_batches['missingness_real'].keys():
            batches = test_batches['missingness_real'][missing_rate]
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
                    dist, log_probs = model.predict(input_batch)
                    nll = -log_probs.sum(-1).mean().item()
                    mean_nll += nll

                tnpd_nlls_missingness_real[missing_rate].append(mean_nll / config['num_available_features'])

        # actions_dict = {subgroup: {} for subgroup in test_batches['subgroups'].keys()}

        # for subgroup in test_batches['subgroups'].keys():
        #     batches = test_batches['subgroups'][subgroup]
        #     for i, batch in enumerate(batches):
        #         # Move batch tensors to the correct device
        #         input_batch = sampler.mask_features(batch, num_observed=0)
        #         input_batch = {k: v.to('cuda') for k, v in input_batch.items()}
        #         mask = input_batch['maskt']

        #         groups = batch['groups']
        #         group_feature_indices = batch['group_feature_indices']
        #         actions = []

        #         mean_nll = 0.0
        #         for j in range(1, config['num_available_features'] + 1):    
        #             outs = model.select_action(input_batch)
        #             action = torch.argmax(outs - 1e6 * mask, dim=-1) # ensure no repeats

        #             input_batch, mask = sampler.acquire_features(batch, mask, action=action)
        #             dist, log_probs = model.predict(input_batch)
        #             nll = -log_probs.sum(-1).mean().item()
        #             mean_nll += nll
        #             actions.append(action)

        #         tnpd_nlls_subgroups[subgroup].append(mean_nll / config['num_available_features'])
                
        #         actions = torch.stack(actions).squeeze(1)
        #         actions_dict[subgroup][f"actions_{i}"] = actions
        #         actions_dict[subgroup][f"groups_{i}"] = groups
        #         actions_dict[subgroup][f"group_feature_indices_{i}"] = group_feature_indices

    avg_losses = []
    avg_losses_missingness = []
    avg_losses_subgroups = []
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

    print("\n==== [TEST] Missingness ICL-based AFA avg_log_loss by missing rate (Real) ====")
    for missing_rate in tnpd_nlls_missingness_real.keys():
        values = tnpd_nlls_missingness_real[missing_rate]
        avg_nll = sum(values) / len(values)
        std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
        print(f"[TEST] ICL-based AFA avg_log_loss (missing_rate={missing_rate}): {avg_nll:.6f} ± {std_err:.6f}")

    # print("\n==== [TEST] Adaptivity ICL-based AFA avg_log_loss by subgroups ====")
    # for subgroup in tnpd_nlls_subgroups.keys():
    #     values = tnpd_nlls_subgroups[subgroup]
    #     avg_nll = sum(values) / len(values)
    #     avg_losses_subgroups.append(avg_nll)
    #     std_err = (sum((x - avg_nll) ** 2 for x in values) / (len(values) * (len(values) - 1))) ** 0.5
    #     print(f"[TEST] ICL-based AFA avg_log_loss (subgroup={subgroup}): {avg_nll:.6f} ± {std_err:.6f}")

    results = {
        "Unseen": avg_losses,
        "Unseen_nlls": tnpd_nlls,
        "Missingness": avg_losses_missingness,
        "Missingness_nlls": tnpd_nlls_missingness,
        "Missingness_nlls_real": tnpd_nlls_missingness_real, 
        "Subgroups": avg_losses_subgroups,
        "Subgroups_nlls": tnpd_nlls_subgroups,
    }
    results_path = os.path.join(log_dir, f"test_results_afa_bench_{config['downstream_task']}.pt")
    torch.save(results, results_path)