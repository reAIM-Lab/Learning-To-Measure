import torch
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from src.data.data_utils import TabularDataset, preprocess_df


def load_mimic_data(config):
    df = pd.read_csv(Path(config['data_dir']) / f"{config['downstream_task']}_mimic.csv")

    features = ['hemoglobin', 'platelet', 'rbc', 'wbc',
                'bun', 'calcium', 'chloride', 'creatinine', 'glucose',
                'rdw', 'age', 'gender', 'icu', 'boolean_value']
    
    cols_to_drop = [col for col in df.columns if col not in features]
    df = df.drop(columns=cols_to_drop)
    df = df.dropna().reset_index(drop=True) # Drop rows with missing values
    df = preprocess_df(df) # One-hot encode categorical columns

    # Get feature names from processed dataframe
    feature_list = list(df.drop(columns=['boolean_value']).columns)
    free_indices = [i for i, col in enumerate(feature_list) if col in ['age', 'gender', 'icu']]

    if config['feature_selection']:
        X = df.drop(columns=['boolean_value']).values
        y = df['boolean_value'].values
        
        # Create and train XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X, y)

        # Get feature importance scores
        importance_scores = xgb_model.feature_importances_
        
        # Create list of (feature, importance) tuples and sort by importance
        feature_importance = list(zip(feature_list, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print("\nTop features from XGBoost:")
        for feat, imp in feature_importance:
            print(f"{feat}: {imp:.4f}")
    
    # Convert dataframe to tensor
    label = torch.tensor(df['boolean_value'].values, dtype=torch.float32)
    x = torch.tensor(df.drop(columns=['boolean_value']).values, dtype=torch.float32)

    if config['margin']:
        dtrain = xgb.DMatrix(x.numpy(), label=label.numpy())
        booster = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 42},
            dtrain,
            num_boost_round=100,
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
        keep_size = config.get("subsample_size", config['test_pool_size'] * 2) # buffer for rounding
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
    rng.shuffle(remaining_indices)
    test_indices = remaining_indices[:config['test_pool_size']]

    # Calculate and print label prevalence
    label_prevalence = torch.mean(label[train_indices]).item()
    print(f"\nLabel prevalence: {label_prevalence:.2%}")
    
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