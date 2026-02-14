# Importing libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

from sklearn.pipeline import Pipeline

import joblib

#Loading data
# url = r"C:\Users\owner\OneDrive\Desktop\ML_projects\Fault_Analysis\merged_dataset.csv"

# url = r"C:\Users\ncc333\Desktop\ML_projects\Fault_Analysis\Main_fault\merged_dataset.csv"

url= r"C:\Users\owner\Desktop\ML_projects\Fault_Analysis\Main_fault\merged_dataset.csv"
def load_data(url):
    df = pd.read_csv(url, sep=",")
    return df

dff = load_data(url)
df = dff.copy()

#droppping time
df= df.drop(['t'], axis=1)


# Preprocessing

# # Feature engineering
# feature_engineer.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window=50, min_periods=1):
        self.window = window
        self.min_periods = min_periods

        # Define columns used
        self.volt_cols = ['Va', 'Vb', 'Vc']
        self.curr_cols = ['Ia', 'Ib', 'Ic']
        self.all_cols = self.volt_cols + self.curr_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Rolling RMS
        for col in self.all_cols:
            df[f"{col}_rms"] = (
                df[col].rolling(self.window, self.min_periods).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
            )

        # Rolling peak-to-peak
        for col in self.all_cols:
            df[f"{col}_ptp"] = (
                df[col].rolling(self.window, self.min_periods).apply(lambda x: np.ptp(x), raw=True)
            )

        # Phase-to-phase voltages
        df['Vab'] = df['Va'] - df['Vb']
        df['Vbc'] = df['Vb'] - df['Vc']
        df['Vca'] = df['Vc'] - df['Va']

        # Phase-to-phase currents
        df['Iab'] = df['Ia'] - df['Ib']
        df['Ibc'] = df['Ib'] - df['Ic']
        df['Ica'] = df['Ic'] - df['Ia']

        # Power
        df['P_total'] = (
            df['Va'] * df['Ia'] +
            df['Vb'] * df['Ib'] +
            df['Vc'] * df['Ic']
        )

        return df