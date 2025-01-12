import pandas as pd
from sklearn.model_selection import train_test_split
from app.core.config import RANDOM_STATE

def load_and_split_data(dataset_path):
    try:
        data = pd.read_csv(dataset_path, encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    required_cols = ["title", "isAntiState"]
    if not all(col in data.columns for col in required_cols):
        missing_cols = set(required_cols) - set(data.columns)
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    anti_state_data = data[data["isAntiState"] == 1]
    normal_data = data[data["isAntiState"] == 0]
    
    train_anti_state, temp_anti_state = train_test_split(anti_state_data, test_size=0.3, random_state=RANDOM_STATE)
    val_anti_state, test_anti_state = train_test_split(temp_anti_state, test_size=0.5, random_state=RANDOM_STATE)
    
    train_normal, temp_normal = train_test_split(normal_data, test_size=0.3, random_state=RANDOM_STATE)
    val_normal, test_normal = train_test_split(temp_normal, test_size=0.5, random_state=RANDOM_STATE)
    
    train_anti_state_labeled = train_anti_state.sample(frac=0.5, random_state=RANDOM_STATE)
    train_normal_labeled = train_normal.sample(frac=0.5, random_state=RANDOM_STATE)
    
    train_labeled = pd.concat([train_anti_state_labeled, train_normal_labeled])
    
    return train_labeled