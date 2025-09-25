import torch
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import random
from torchvision import datasets, transforms

from src.data.data_utils import TabularDataset

def load_mnist_data(config):
    save_path = Path(config['data_dir']) / 'mnist_preprocessed.parquet'

    if save_path.exists():
        # Load preprocessed data
        df = pd.read_parquet(save_path, engine="pyarrow")
        x = df.drop(columns=['label', 'img']).values
        label = df['label'].values
        # feature_list = df.drop(columns=['label']).columns
        feature_list = ['block_0_4', 'block_1_6', 'block_2_2', 'block_3_5', 'block_0_3', 
                        'block_0_2', 'block_5_2', 'block_4_6', 'block_5_0', 'block_4_2', 
                        'block_4_3', 'block_5_6', 'block_6_3', 'block_3_3', 'block_1_3', 
                        'block_5_1', 'block_4_4', 'block_3_2', 'block_5_5', 'block_2_4']
        print(f"Loaded preprocessed data from {save_path}")
    
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] -> [0,1] float tensor
            transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
        ])

        # Training + test datasets
        train_dataset = datasets.MNIST(root=config['data_dir'], train=True, transform=transform, download=True)
        block_size = 4  # 4x4 pixel blocks
        num_blocks = 28 // block_size  # 7 blocks per dimension

        feature_list = [f"block_{i}_{j}" for i in range(num_blocks) for j in range(num_blocks)]

        X_list, y_list, img_list = [], [], []
        for img, y in train_dataset:
            img_np = img.squeeze().numpy()  # [28,28]
            
            blocks = []
            for i in range(num_blocks):
                for j in range(num_blocks):
                    block = img_np[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    blocks.append(block.mean())  # average pooling
            X_list.append(np.array(blocks))
            y_list.append(y)
            img_list.append(img_np)

        x = np.stack(X_list)  # shape [num_samples, 784]
        label = np.array(y_list)

        if config['feature_selection']:        
            # Create and train XGBoost model
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(x, label)

            # Get feature importance scores
            importance_scores = xgb_model.feature_importances_
            
            # Create list of (feature, importance) tuples and sort by importance
            feature_importance = list(zip(feature_list, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            top_features = feature_importance[:20]
            top_feature_names = [feat for feat, imp in top_features]

            print("\nTop features from XGBoost:")
            for feat, imp in feature_importance:
                print(f"{feat}: {imp:.4f}")

            print(top_feature_names)

            top_indices = [feature_list.index(feat) for feat in top_feature_names]
            x = x[:, top_indices]

            df = pd.DataFrame(x, columns=top_feature_names)
            df['label'] = label
            df["img"] = [img.astype("float32").tolist() for img in img_list]
            df.to_parquet(save_path, engine="pyarrow", index=False)

    x = torch.tensor(x, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)

    # Generate indices for train/test split
    num_samples = len(x)
    indices = np.arange(num_samples)

    rng = np.random.RandomState(config['data_seed'])
    rng.shuffle(indices)
    
    # Take first config['num_points'] samples for context during evaluation
    train_indices = indices[:config['pool_size']]

    # Take pool_size targets points for evaluation
    remaining_indices = np.setdiff1d(indices, train_indices)
    rng.shuffle(remaining_indices)
    test_indices = remaining_indices[:config['pool_size']]

    # Calculate and print label prevalence
    # label_prevalence = torch.mean(label[train_indices]).item()
    # print(f"\nLabel prevalence: {label_prevalence:.2%}")

    # label_prevalence = torch.mean(label[test_indices]).item()
    # print(f"\nTest label prevalence: {label_prevalence:.2%}")
    
    # Create train/test datasets
    train_dataset = TabularDataset(
        data=x[train_indices],
        dim_y=config['label_dim'],
        feature_names=feature_list,
        dataframe=df.iloc[train_indices].reset_index(drop=True),
        free_indices=[],
        label=label[train_indices]
    )
    
    test_dataset = TabularDataset(
        data=x[test_indices], 
        dim_y=config['label_dim'],
        feature_names=feature_list,
        dataframe=df.iloc[test_indices].reset_index(drop=True),
        free_indices=[],
        label=label[test_indices]
    )

    return train_dataset, test_dataset

class MNISTSampler(object):
    def __init__(self, train_dataset, test_dataset, seed=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dim_x = train_dataset.feature_dim
        self.dim_y = train_dataset.dim_y
        self.free_indices = train_dataset.free_indices
        self.feature_groups = train_dataset.feature_groups
        self.num_available_features = train_dataset.num_available_features

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed

    @staticmethod
    def _sample_indices(dataset: TabularDataset, size: int, batch_size: int, seed=None) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            return torch.stack([
                torch.randperm(len(dataset), generator=generator)[:size]
            for _ in range(batch_size)
        ])
        else:
            return torch.stack([
                torch.randperm(len(dataset))[:size]
            for _ in range(batch_size)
        ])
    
    @staticmethod
    def _normalize(x):
        # Normalize features
        mean = x.mean(dim=1, keepdim=True)     # [B, 1, Dx]
        std = x.std(dim=1, keepdim=True) + 1e-8  # [B, 1, Dx] (avoid div by zero)
        x = (x - mean) / std
        return x, mean, std
    
    def _generate_missingness(self, x, num_observed=None, seed=None, missing_rate=None):
        """Generate random missingness mask for each sample in batch.
        """
        batch_size, seq_len, num_features = x.shape
        
        if num_observed is None:
            # Randomly mask features
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
                unif = torch.rand(batch_size, seq_len, num_features, generator=generator)
                if missing_rate is not None:
                    ref = torch.full((batch_size, seq_len, 1), missing_rate)
                else:
                    ref = torch.rand(batch_size, seq_len, 1, generator=generator)
            else:
                unif = torch.rand(batch_size, seq_len, num_features)
                if missing_rate is not None:
                    ref = torch.full((batch_size, seq_len, 1), missing_rate)
                else:   
                    ref = torch.rand(batch_size, seq_len, 1)
                    
            mask = (unif > ref).float().to(x.device)
        else:
            if num_observed == 0:
                mask = torch.zeros_like(x)
            elif num_observed == num_features:
                mask = torch.ones_like(x)
            else:
                # Handle free indices if they exist
                if self.free_indices is not None:
                    
                    # Only randomly mask non-free indices
                    non_free_indices = [i for i in range(num_features) if i not in self.free_indices]
                    non_free_indices = torch.tensor(non_free_indices, device=x.device)
                    observed_features = torch.empty(batch_size, seq_len, num_observed, dtype=torch.long, device=x.device)
                    for b in range(batch_size):
                        for t in range(seq_len):
                            perm = torch.randperm(len(non_free_indices), device=x.device)[:num_observed]
                            observed_features[b, t] = non_free_indices[perm]
                    # Update mask with randomly selected features
                    mask = torch.zeros_like(x)
                    mask.scatter_(2, observed_features, 1.0)
                else:
                    rand_scores = torch.rand(batch_size, seq_len, num_features, device=x.device)
                    _, observed_features = torch.topk(rand_scores, num_observed, dim=2)
                    mask = torch.zeros_like(x)
                    mask.scatter_(2, observed_features, 1.0)

        if self.free_indices is not None:
            mask[:, :, self.free_indices] = 1.0
                
        return mask

    def sample(self, batch_size=16, num_ctx=None, num_tar=None, max_num_points=1000, device='cpu', num_observed=None, seed=None, return_groups=False):

        batch = {}
        num_ctx = num_ctx or 50
        num_tar = max_num_points - num_ctx
        num_points = num_ctx + num_tar  # N = Nc + Nt

        all_classes = torch.unique(self.train_dataset.label).tolist()
        batch_x = []
        batch_y = []

        for _ in range(batch_size):
            # Pick 2 digits for this sequence
            chosen_classes = random.sample(all_classes, 2)

            mask = torch.isin(self.train_dataset.label, torch.tensor(chosen_classes))
            filtered_data = self.train_dataset.data[mask]
            filtered_labels = self.train_dataset.label[mask]

            label_map = {chosen_classes[0]: 0.0, chosen_classes[1]: 1.0}
            binary_labels = torch.tensor([label_map[int(lbl)] for lbl in filtered_labels])

            filtered_dataset = TabularDataset(
                data=filtered_data,
                dim_y=self.dim_y,
                feature_names=self.train_dataset.feature_names,
                dataframe=None,
                free_indices=[],
                label=binary_labels,
            )

            # Sample a sequence from this filtered dataset
            indices = self._sample_indices(filtered_dataset, num_points, batch_size=1, seed=seed)
            batch_x.append(filtered_dataset.data[indices])   # shape [1, N, Dx]
            batch_y.append(filtered_dataset.label[indices])  # shape [1, N]

        # Stack into batch
        x = torch.cat(batch_x, dim=0)   # [B, N, Dx]
        y = torch.cat(batch_y, dim=0)

        # Normalize data
        x, _, _ = self._normalize(x)
        
        # Sample retrospective MAR missingness mechanisms
        missing_probs = torch.rand(batch_size, 1, self.dim_x) * 0.5  # [B,1,Dx]
        rand_vals = torch.rand([batch_size, num_points, self.dim_x])
        r = (rand_vals > missing_probs).float()

        # Generate target state missingness mask: 1 = observed, 0 = missing
        maskt = self._generate_missingness(x[:, num_ctx:], num_observed=num_observed)
        mask = torch.ones_like(x[:, :, :self.dim_x])
        mask[:,num_ctx:,:] *= maskt 
        mask *= r
        
        batch['x'] = x.to(device)
        batch['xc'] = batch['x'][:, :num_ctx]  # [B,Nc,2*Dx]
        batch['xt'] = batch['x'][:, num_ctx:]  # [B,Nt,2*Dx]
        
        batch['y'] = y.unsqueeze(-1).to(device)
        batch['yc'] = batch['y'][:, :num_ctx]  # [B,Nc,1]
        batch['yt'] = batch['y'][:, num_ctx:]  # [B,Nt,1]

        batch['mask'] = mask.to(device)
        batch['maskc'] = batch['mask'][:, :num_ctx]
        batch['maskt'] = batch['mask'][:, num_ctx:]

        batch['r'] = r.to(device)
        batch['rc'] = batch['r'][:, :num_ctx]
        batch['rt'] = batch['r'][:, num_ctx:]

        return batch
    
    def sample_test(self, 
                    batch_size=16, 
                    num_ctx=None, 
                    num_tar=None, 
                    max_num_points=1000, 
                    device='cpu', 
                    num_observed=None, 
                    max_p_missing=0.5,
                    unseen=True,
                    real_y=True,
                    min_informative=1,
                    digits=None,
                    return_images=False,
                    seed=None):
        batch = {}

        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            num_ctx = num_ctx or torch.randint(low=50, high=max_num_points-50, size=[1], generator=g).item()  # Nc
            num_tar = num_tar or torch.randint(low=50, high=max_num_points-num_ctx, size=[1], generator=g).item()  # Nt
        else:
            num_ctx = num_ctx or torch.randint(low=50, high=max_num_points-50, size=[1]).item()  # Nc
            num_tar = num_tar or torch.randint(low=50, high=max_num_points-num_ctx, size=[1]).item()  # Nt

        num_points = num_ctx + num_tar  # N = Nc + Nt
        # Sample random context sequence from train dataset and target sequence from test dataset
        all_classes = torch.unique(self.test_dataset.label).tolist()
        batch_x = []
        batch_y = []
        batch_dfs = []

        for _ in range(batch_size):
            # Pick 2 digits for this sequence if not provided
            chosen_classes = random.sample(all_classes, 2)

            mask = torch.isin(self.test_dataset.label, torch.tensor(chosen_classes))
            filtered_data = self.test_dataset.data[mask]
            filtered_labels = self.test_dataset.label[mask]
            df_filtered = self.test_dataset.dataframe.loc[mask.numpy()].reset_index(drop=True)

            label_map = {chosen_classes[0]: 0.0, chosen_classes[1]: 1.0}
            binary_labels = torch.tensor([label_map[int(lbl)] for lbl in filtered_labels])

            filtered_dataset = TabularDataset(
                data=filtered_data,
                dim_y=self.dim_y,
                feature_names=self.test_dataset.feature_names,
                dataframe=df_filtered,
                free_indices=[],
                label=binary_labels,
            )

            # Sample a sequence from this filtered dataset
            indices = self._sample_indices(filtered_dataset, num_points, batch_size=1, seed=seed)
            batch_x.append(filtered_dataset.data[indices])   # shape [1, N, Dx]
            batch_y.append(filtered_dataset.label[indices])  # shape [1, N]
            batch_dfs.append(df_filtered.iloc[indices.view(-1).tolist()].reset_index(drop=True))

        # Stack into batch
        x = torch.cat(batch_x, dim=0)   # [B, N, Dx]
        y = torch.cat(batch_y, dim=0)

        # Normalize data
        x_ctx, mean, std = self._normalize(x[:, :num_ctx])
        x_tar = (x[:, num_ctx:] - mean) / std
        x = torch.cat([x_ctx, x_tar], dim=1)
        
        # Sample missingness mechanisms
        missing_probs = torch.rand(batch_size, 1, self.dim_x) * max_p_missing  # [B,1,Dx]
        rand_vals = torch.rand([batch_size, num_points, self.dim_x])
        r = (rand_vals > missing_probs).float()

        # Target fully observed for evaluation
        r[:, num_ctx:] = 1.0

        mask = torch.ones_like(x[:, :, :self.dim_x])
        maskt = self._generate_missingness(x[:, num_ctx:], num_observed)
        mask[:,num_ctx:,:] *= maskt 

        batch['x'] = x.to(device)
        batch['xc'] = batch['x'][:, :num_ctx]  # [B,Nc,2*Dx]
        batch['xt'] = batch['x'][:, num_ctx:]  # [B,Nt,2*Dx]
        
        batch['y'] = y.unsqueeze(-1).to(device)
        batch['yc'] = batch['y'][:, :num_ctx]  # [B,Nc,1]
        batch['yt'] = batch['y'][:, num_ctx:]  # [B,Nt,1]

        batch['mask'] = mask.to(device)
        batch['maskc'] = batch['mask'][:, :num_ctx]
        batch['maskt'] = batch['mask'][:, num_ctx:]

        batch['r'] = r.to(device)
        batch['rc'] = batch['r'][:, :num_ctx]
        batch['rt'] = batch['r'][:, num_ctx:]

        if return_images:
            return batch, batch_dfs
        
        return batch
    
    def sample_context(self, batch, num_ctx):
        new_ctx = batch['xc'][:, :num_ctx].clone()
        new_ctx_y = batch['yc'][:, :num_ctx].clone()
        new_ctx_mask = batch['maskc'][:, :num_ctx].clone()
        new_ctx_r = batch['rc'][:, :num_ctx].clone()

        x = torch.cat([new_ctx, batch['xt']], dim=1)

        new_batch = {}
        new_batch['x'] = x
        new_batch['xc'] = new_ctx  # [B,Nc,2*Dx]
        new_batch['xt'] = batch['xt']

        new_batch['y'] = batch['y']
        new_batch['yc'] = new_ctx_y
        new_batch['yt'] = batch['yt']

        new_batch['mask'] = torch.cat([new_ctx_mask, batch['maskt']], dim=1)
        new_batch['maskc'] = new_ctx_mask
        new_batch['maskt'] = batch['maskt']

        new_batch['r'] = batch['r']
        new_batch['rc'] = new_ctx_r
        new_batch['rt'] = batch['rt']

        return new_batch
    
    def mask_features(self, batch, num_observed=None, mask=None, seed=None, missing_rate=None):
        x = batch['xt']

        # Sample target state missingness mask if not provided
        if mask is None:
            mask = self._generate_missingness(x, num_observed=num_observed, seed=seed, missing_rate=missing_rate)

        new_batch = {
            'x': batch['x'],
            'xc': batch['xc'],
            'xt': batch['xt'],
            'y': batch['y'],
            'yc': batch['yc'],
            'yt': batch['yt'],
            'mask': torch.cat((batch['maskc'], mask), dim=1),
            'maskc': batch['maskc'],
            'maskt': mask,
            'r': batch['r'],
            'rc': batch['rc'],
            'rt': batch['rt']
        }

        return new_batch
    
    def initialize_mask(self, batch):
        maskt = torch.zeros_like(batch['maskt'])

        if self.free_indices is not None:
            maskt[:, :, self.free_indices] = 1.0
        
        return maskt
    
    def acquire_features(self, batch, mask, action=None, seed=None):
        batch_size, seq_len, dim_x = batch['maskt'].shape
        xt = batch['xt']
        rt = batch['rt'] if 'rt' in batch else None

        # If mask is not provided, initialize it to all zeros
        if mask is None:
            mask = torch.zeros_like(batch['maskt'])

            if self.free_indices is not None:
                mask[:, :, self.free_indices] = 1.0

        # If action is not provided, randomly select an action
        if action is None:
            # Create new mask
            new_mask = mask.clone()

            if rt is not None:
                available_mask = (mask == 0) & (rt == 1)  # shape: (B, T, F)
            else:
                available_mask = (mask == 0)

            B, T, F = available_mask.shape

            if seed is not None:
                g = torch.Generator(device=available_mask.device).manual_seed(seed)
                logits = torch.randn(B, T, F, generator=g, device=available_mask.device)
            else:
                logits = torch.randn(B, T, F, device=available_mask.device)

            masked_logits = logits.masked_fill(~available_mask, -1e9)
            idx = masked_logits.argmax(dim=-1)

            new_mask.scatter_(-1, idx.unsqueeze(-1), 1.0)

        else:
            new_mask = mask.clone()
            batch_idx = torch.arange(batch_size, device=action.device)[:, None]    # shape [B, 1]
            time_idx  = torch.arange(seq_len, device=action.device)[None, :]     # shape [1, T]
            valid = rt[batch_idx, time_idx, action] == 1
            new_mask[batch_idx, time_idx, action] = valid.float()

        #xt_input = self._compose_inputs(xt, new_mask)  # [B,N,2*Dx]

        new_batch = {}
        new_batch['x'] = batch['x']
        new_batch['xc'] = batch['xc']  # [B,Nc,2*Dx]
        new_batch['xt'] = batch['xt']

        new_batch['y'] = batch['y']
        new_batch['yc'] = batch['yc']
        new_batch['yt'] = batch['yt']

        new_batch['mask'] = torch.cat([batch['maskc'], new_mask], dim=1)
        new_batch['maskc'] = batch['maskc']
        new_batch['maskt'] = new_mask

        new_batch['r'] = batch['r']
        new_batch['rc'] = batch['rc']
        new_batch['rt'] = batch['rt']
    
        return new_batch, new_mask

