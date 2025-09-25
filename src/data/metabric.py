import torch
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from src.data.data_utils import TabularDataset, preprocess_df


def load_metabric_data(config):
    df = pd.read_csv(Path(config['data_dir']) / "METABRIC_RNA_Mutation.csv", low_memory=False)

    features = ['ccnb1', 'cdk1', 'e2f2',
                'e2f7', 'stat5b', 'notch1', 'rbpj', 'bcl2', 'egfr', 'erbb2', 'erbb3', 'abcb1'
                  'age_at_diagnosis', 'pam50_+_claudin-low_subtype']
    
    df = df[df['pam50_+_claudin-low_subtype'] != "NC"].copy()
    
    cols_to_drop = [col for col in df.columns if col not in features]
    df = df.drop(columns=cols_to_drop)
    df = df.dropna().reset_index(drop=True) # Drop rows with missing values
    df = preprocess_df(df) # Standardize continuous columns and one-hot encode categorical columns

    # Get feature names from processed dataframe
    feature_list = list(df.columns)
    free_indices = [i for i, col in enumerate(feature_list) if col in ['age_at_diagnosis']]

    # Convert dataframe to tensor
    label = torch.tensor(df[[c for c in feature_list if c.startswith("pam50")]].values, dtype=torch.float32)
    x = torch.tensor(df.drop(columns=[c for c in feature_list if c.startswith("pam50")]).values, dtype=torch.float32)

    # label_classes = label.argmax(dim=-1)
    # counts_b = torch.bincount(label_classes, minlength=config['label_dim'])
    # prev = counts_b.float() / label_classes.numel()
    # print("Prevalence in y[0]:", ", ".join([f"class {k}: {p:.2%}" for k,p in enumerate(prev)]))

    if config['margin']:
        dtrain = xgb.DMatrix(x.numpy(), label=label.numpy())
        booster = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "logloss"},
            dtrain,
            num_boost_round=100
        )

        probs = booster.predict(dtrain)
        margins = np.abs(probs - 0.5)

        pos_indices = np.where(label.numpy() == 1)[0]
        neg_indices = np.where(label.numpy() == 0)[0]

        pos_margins = margins[pos_indices]
        neg_margins = margins[neg_indices]

        # rank within each class
        pos_sorted = pos_indices[np.argsort(-pos_margins)]
        neg_sorted = neg_indices[np.argsort(-neg_margins)]

        # pick equally
        keep_size = config.get("subsample_size", config['pool_size'] * 2 + 10) # buffer for rounding
        prevalence = len(pos_indices) / len(label)
        pos_keep = int(round(keep_size * prevalence))
        neg_keep = keep_size - pos_keep  # remainder goes to negatives

        selected_indices = np.concatenate([pos_sorted[:pos_keep], neg_sorted[:neg_keep]])

        x = x[selected_indices]
        label = label[selected_indices]

    # Generate indices for train/test split
    num_samples = len(x)
    indices = np.arange(num_samples)

    rng = np.random.RandomState(config['data_seed'])
    rng.shuffle(indices)
    
    # Take first config['num_points'] samples for context during evaluation
    train_indices = indices[config['num_points']:config['num_points'] + config['pool_size']]

    # Take pool_size targets points for evaluation
    remaining_indices = np.setdiff1d(indices, train_indices)
    test_indices = remaining_indices[:config['pool_size']]
    
    # Create train/test datasets
    train_dataset = TabularDataset(
        data=x[train_indices],
        dim_y=config['label_dim'],
        feature_names=feature_list,
        dataframe=df.iloc[train_indices],
        free_indices=free_indices,
        label=label[train_indices]
    )
    
    test_dataset = TabularDataset(
        data=x[test_indices], 
        dim_y=config['label_dim'],
        feature_names=feature_list,
        dataframe=df.iloc[test_indices],
        free_indices=free_indices,
        label=label[test_indices]
    )

    return train_dataset, test_dataset