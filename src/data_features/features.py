from data_loader.data_loader_2 import  load_data
import pandas as pd
import numpy as np

from data_loader.data_loader_2 import load_data
import pandas as pd
import numpy as np

class FeatureCreation:
    def __init__(self, data_path):
        self.data = load_data(data_path)

    def create_features(self):
       
        if 'trans_date_trans_time' not in self.data.columns:
            raise ValueError("Column 'trans_date_trans_time' not found in data.")
        
        self.data['trans_date_trans_time'] = pd.to_datetime(self.data['trans_date_trans_time'], errors='coerce')
        self.data['hour'] = self.data['trans_date_trans_time'].dt.hour
        self.data['month'] = self.data['trans_date_trans_time'].dt.month
        self.data['day'] = self.data['trans_date_trans_time'].dt.day
        self.data['dayofweek'] = self.data['trans_date_trans_time'].dt.dayofweek
        self.data['is_weekend'] = self.data['dayofweek'] >= 5
        
        return self.data
    def check_missing_values(self):
        if self.data.isnull().sum() > 0:
            raise ValueError("Data contains missing values.")
        else:
            print("No missing values found in the data.")

    



