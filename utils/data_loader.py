import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

class WindPowerDataset(Dataset):
    def __init__(self, csv_file, wind_power_scaler=None, weather_scaler=None, save_scalers=False):
        self.data = pd.read_csv(csv_file)
         
        self.data = self.data.iloc[::5, :].reset_index(drop=True)
        
        self.original_wind_power_std = self.data.iloc[:, 2].std()
        self.original_weather_std = self.data.iloc[:, 4:12].std()
        
        if wind_power_scaler is None or weather_scaler is None:
            self.wind_power_scaler = MinMaxScaler()
            self.weather_scaler = MinMaxScaler()

            self.data.iloc[:, 2] = self.wind_power_scaler.fit_transform(self.data.iloc[:, 2].values.reshape(-1, 1)).squeeze()

            self.data.iloc[:, 4:12] = self.weather_scaler.fit_transform(self.data.iloc[:, 4:12])
            
            if save_scalers:
                with open('wind_power_scaler.pkl', 'wb') as f:
                    pickle.dump(self.wind_power_scaler, f)
                with open('weather_scaler.pkl', 'wb') as f:
                    pickle.dump(self.weather_scaler, f)
        else:
            self.wind_power_scaler = wind_power_scaler
            self.weather_scaler = weather_scaler
            self.data.iloc[:, 2] = self.wind_power_scaler.transform(self.data.iloc[:, 2].values.reshape(-1, 1)).squeeze()
            self.data.iloc[:, 4:12] = self.weather_scaler.transform(self.data.iloc[:, 4:12])
    
    def __len__(self):
        return len(self.data) - 312  # 288 (1440/5) + 24 (120/5)
    
    def __getitem__(self, idx):
        history_weather = self.data.iloc[idx:idx + 288, 4:12].values.astype(float)  # [288, 8]
        wind_power_history = self.data.iloc[idx:idx + 288, 2].values.astype(float)   # [288]
        future_weather = self.data.iloc[idx + 288:idx + 312, 4:12].values.astype(float)  # [24, 8]
        future_wind_power = self.data.iloc[idx + 312, 2]  # float

        return (
            torch.tensor(history_weather, dtype=torch.float32),
            torch.tensor(wind_power_history, dtype=torch.float32),
            torch.tensor(future_weather, dtype=torch.float32),
            torch.tensor(future_wind_power, dtype=torch.float32)
        )
    
    def get_original_stds(self):
        return {
            'original_wind_power_std': self.original_wind_power_std,
            'original_weather_std': self.original_weather_std.to_dict()
        }
    

class WindPowerDataset(Dataset):
    def __init__(self, csv_file, wind_power_scaler=None, weather_scaler=None, save_scalers=False):
        self.data = pd.read_csv(csv_file)

        self.data = self.data.iloc[::5, :].reset_index(drop=True)

        self.original_wind_power_std = self.data.iloc[:, 2].std()
        self.original_weather_std = self.data.iloc[:, 4:12].std()

        if wind_power_scaler is None or weather_scaler is None:
            self.wind_power_scaler = MinMaxScaler()
            self.weather_scaler = MinMaxScaler()

            self.data.iloc[:, 2] = self.wind_power_scaler.fit_transform(self.data.iloc[:, 2].values.reshape(-1, 1)).squeeze()

            self.data.iloc[:, 4:12] = self.weather_scaler.fit_transform(self.data.iloc[:, 4:12])
            
            if save_scalers:
                with open('wind_power_scaler_PJM_zone_1_.pkl', 'wb') as f:
                    pickle.dump(self.wind_power_scaler, f)
                with open('weather_scaler_PJM_zone_1_.pkl', 'wb') as f:
                    pickle.dump(self.weather_scaler, f)
        else:
            self.wind_power_scaler = wind_power_scaler
            self.weather_scaler = weather_scaler
            self.data.iloc[:, 2] = self.wind_power_scaler.transform(self.data.iloc[:, 2].values.reshape(-1, 1)).squeeze()
            self.data.iloc[:, 4:12] = self.weather_scaler.transform(self.data.iloc[:, 4:12])
    
    def __len__(self):
        return len(self.data) - 312  # 288 (1440/5) + 24 (120/5)
    
    def __getitem__(self, idx):
        history_weather = self.data.iloc[idx:idx + 288, 4:12].values.astype(float)  # [288, 8]
        wind_power_history = self.data.iloc[idx:idx + 288, 2].values.astype(float)   # [288]
        future_weather = self.data.iloc[idx + 288:idx + 312, 4:12].values.astype(float)  # [24, 8]
        future_wind_power = self.data.iloc[idx + 312, 2]  # float

        return (
            torch.tensor(history_weather, dtype=torch.float32),
            torch.tensor(wind_power_history, dtype=torch.float32),
            torch.tensor(future_weather, dtype=torch.float32),
            torch.tensor(future_wind_power, dtype=torch.float32)
        )
    
    def get_original_stds(self):
        return {
            'original_wind_power_std': self.original_wind_power_std,
            'original_weather_std': self.original_weather_std.to_dict()
        }
    

class WindPowerDataset(Dataset):
    def __init__(self, csv_file, wind_power_scaler=None, weather_scaler=None, save_scalers=False):
        self.data = pd.read_csv(csv_file)

        self.data = self.data.iloc[::5, :].reset_index(drop=True)

        self.original_wind_power_std = self.data.iloc[:, 2].std()
        self.original_weather_std = self.data.iloc[:, 4:12].std()

        if wind_power_scaler is None or weather_scaler is None:
            self.wind_power_scaler = MinMaxScaler()
            self.weather_scaler = MinMaxScaler()

            self.data.iloc[:, 2] = self.wind_power_scaler.fit_transform(self.data.iloc[:, 2].values.reshape(-1, 1)).squeeze()

            self.data.iloc[:, 4:12] = self.weather_scaler.fit_transform(self.data.iloc[:, 4:12])
            
            if save_scalers:
                with open('wind_power_scaler_MISO_zone_1_.pkl', 'wb') as f:
                    pickle.dump(self.wind_power_scaler, f)
                with open('weather_scaler_MISO_zone_1_.pkl', 'wb') as f:
                    pickle.dump(self.weather_scaler, f)
        else:
            self.wind_power_scaler = wind_power_scaler
            self.weather_scaler = weather_scaler
            self.data.iloc[:, 2] = self.wind_power_scaler.transform(self.data.iloc[:, 2].values.reshape(-1, 1)).squeeze()
            self.data.iloc[:, 4:12] = self.weather_scaler.transform(self.data.iloc[:, 4:12])
    
    def __len__(self):
        return len(self.data) - 312  # 288 (1440/5) + 24 (120/5)
    
    def __getitem__(self, idx):
        history_weather = self.data.iloc[idx:idx + 288, 4:12].values.astype(float)  # [288, 8]
        wind_power_history = self.data.iloc[idx:idx + 288, 2].values.astype(float)   # [288]
        future_weather = self.data.iloc[idx + 288:idx + 312, 4:12].values.astype(float)  # [24, 8]
        future_wind_power = self.data.iloc[idx + 312, 2]  # float

        return (
            torch.tensor(history_weather, dtype=torch.float32),
            torch.tensor(wind_power_history, dtype=torch.float32),
            torch.tensor(future_weather, dtype=torch.float32),
            torch.tensor(future_wind_power, dtype=torch.float32)
        )
    
    def get_original_stds(self):
        return {
            'original_wind_power_std': self.original_wind_power_std,
            'original_weather_std': self.original_weather_std.to_dict()
        }