import pandas as pd

def load_data():
    train_obs = pd.read_csv("data/train_obs.csv")
    train_char = pd.read_csv("data/train_char.csv")
    test_obs = pd.read_csv("data/test_obs.csv")
    test_char = pd.read_csv("data/test_char.csv")
    return train_obs, train_char, test_obs, test_char
