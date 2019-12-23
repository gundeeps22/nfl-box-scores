import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import re

class NFLWeatherDataSet(Dataset):

    def __init__(self):
        data = preprocessing()
        data64 = data.astype("float64")
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data64[:, 1:-1])
        self.y_data = torch.from_numpy(data64[:, [0]])

    def __getitem__ (self, index):
        return self.x_data[index], self.y_data[index]

    def __len__ (self):
        return self.len


def preprocessing():
    df = pd.read_csv('nfl_game_weather.csv')
    df = df.dropna(thresh=df.shape[1]) # drop all rows with only NaN

    # drop columns weather
    df = df.drop(columns = "weather")
    df = df.drop(columns = 'id')
    df = df.drop(columns = 'home_team')
    df = df.drop(columns = 'away_team')
    df = df.drop(columns = 'home_score')
    df = df.drop(columns = 'away_score')
    df = df.drop(columns = 'date')
    # drop dates preceding 2010
    # df = df.drop(df[int(df['date'].split("/")[2]) < 2009].index, inplace = True)
    # convert to numpy array
    np_arry = df.to_numpy()

    for i in range(len(np_arry)):
        np_arry[i][2] = int(np_arry[i][2].replace('%',''))/100.0

    return np_arry

def create_data_loader():
    dataset = NFLWeatherDataSet()
    print(dataset.x_data[0])
    print(dataset.y_data[0])

    dataset_len = len(dataset)
    indexes = list(range(dataset_len))

    split_pt = int(np.floor(0.3 * dataset_len))
    np.random.seed(22)
    np.random.shuffle(indexes)

    train_indexes, valid_indexes = indexes[split_pt:], indexes[:split_pt]

    train_sampling = SubsetRandomSampler(train_indexes)
    valid_sampling = SubsetRandomSampler(valid_indexes)

    train_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=True,
                                sampler=train_sampling)

    valid_loader = DataLoader(dataset=dataset=, batch_size=100, shuffle=True,
                                sampler=valid_sampling)



create_data_loader()

# print(df.head(15))f
