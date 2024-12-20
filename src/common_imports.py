# common_imports.py

"""
Common imports for the ML/AI/Time-Series project.
Ensure that all required libraries are installed:
pip install -r requirements.txt
"""

# Standard libraries
import os
import sys
import json
import datetime
import time
import warnings
import zipfile

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, f1_score, precision_score, 
    recall_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# Torchvision imports
import torchvision
from torchvision import datasets, models, transforms

# Time series specific libraries
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf

# Environment management
from dotenv import load_dotenv

# Other
from collections import Counter