#   NOTE:-
#   For a details of experiments with different models and performance comparison, refer to the 'Modelling_experiments.ipynb'
#   In this section, we will just train the top performing models from each section and save them into Models/:-
#       1. Best performing SVM
#       2. 2 best performing tree-based ensembles
#       3. Best performing Neural Network architecture

#   We will NOT train the Gaussian Process (GP) based model. The main objective of experimenting with that was to 
#   understand the feature importance. This could have been done using Random Forest as well; yet we chose GP because
#   the data distributions of numerical features were largely Gaussian in nature

#   Models will be saved as a .pkl file. They will also be used for inference using the inference.py
#   This section is mainly if you want to train the models again from scratch.


import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier


import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader


#   Used for models like SVM etc...
def model_fit_and_save(X_train, y_train, model, MODEL_SAVE_PATH, use_poly=False, degree=2):
    steps = []

    if use_poly:
        steps.append(('poly', PolynomialFeatures(degree=degree)))

    steps.append(('model', model))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)


    model_name = model.__class__.__name__
    if isinstance(model, SVC):
        model_name += f'_kernel_{model.kernel}'
        if model.kernel == 'poly':
            model_name += f'_deg_{model.degree}'
    if use_poly:
        model_name += f'_polyfeat_deg_{degree}'

    filename = os.path.join(MODEL_SAVE_PATH, f'{model_name}.pkl')


    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)





def save_simple_models(X_train, y_train, MODEL_SAVE_PATH):
    # 1. Logistic Regression
    model_fit_and_save(X_train, y_train,
                       LogisticRegression(max_iter=1000),
                       MODEL_SAVE_PATH, use_poly=False)

    # 2. SVC with linear kernel
    model_fit_and_save(X_train, y_train,
                       SVC(kernel='linear', probability=True),
                       MODEL_SAVE_PATH, use_poly=False)

    # 3. SVC with poly kernel, degree=2
    model_fit_and_save(X_train, y_train,
                       SVC(kernel='poly', degree=2, probability=True),
                       MODEL_SAVE_PATH, use_poly=False)



def train_and_save_tree_model(X_train, y_train, model, model_name, MODEL_SAVE_PATH):
    model.fit(X_train, y_train)

    file_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)



def save_top_tree_models(X_train, y_train, MODEL_SAVE_PATH):
    # 1. CatBoost
    model = CatBoostClassifier(depth=3, iterations=250, learning_rate=0.05, verbose=0, random_state=42)
    train_and_save_tree_model(X_train, y_train, model, 'catboost_depth3_iter250_lr0.05', MODEL_SAVE_PATH)

    # 2. XGBoost
    model = xgb.XGBClassifier(max_depth=3, n_estimators=250, learning_rate=0.05, eval_metric='logloss', random_state=42)
    train_and_save_tree_model(X_train, y_train, model, 'xgboost_depth3_estimators250_lr0.05', MODEL_SAVE_PATH)

    # 3. Random Forest
    model = RandomForestClassifier(max_depth=7, n_estimators=250, random_state=42)
    train_and_save_tree_model(X_train, y_train, model, 'randomforest_depth7_estimators250', MODEL_SAVE_PATH)

    # 4. LightGBM
    model = lgb.LGBMClassifier(max_depth=3, n_estimators=250, learning_rate=0.05, force_col_wise=True, verbosity=-1, random_state=42)
    train_and_save_tree_model(X_train, y_train, model, 'lightgbm_depth3_estimators250_lr0.05', MODEL_SAVE_PATH)



class Network_v1(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)
    


def train_and_save_neural_net_model(X_train, y_train, model, optimizer, MODEL_SAVE_PATH, epochs=15, batch_size=64, device='cpu'):
    model.to(device)
    model.train()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


    model_name = f"{model.__class__.__name__}_binary.pt"
    save_path = os.path.join(MODEL_SAVE_PATH, model_name)
    torch.save(model.state_dict(), save_path)




def save_all_models(final_df_qt, final_df_not_qt, MODEL_SAVE_PATH):
    
    save_simple_models(final_df_qt.drop('target', axis=1), final_df_qt['target'], MODEL_SAVE_PATH)
    save_top_tree_models(final_df_not_qt.drop('target', axis=1), final_df_not_qt['target'], MODEL_SAVE_PATH)
    
    net_model = Network_v1()
    train_and_save_neural_net_model(final_df_qt.drop('target', axis=1).values, final_df_qt['target'].values, 
                                    net_model, AdamW(net_model.parameters(), lr=1e-3), MODEL_SAVE_PATH)




if __name__ == '__main__':
    env_path = Path(os.getcwd()) / 'Config' / '.env'
    load_dotenv(env_path)

    PREPROCESSED_PATH = os.getenv('PREPROCESSED_PATH')
    MODEL_SAVE_PATH = os.getenv('MODEL_PATH')

    final_df_qt = pd.read_csv(os.path.join(PREPROCESSED_PATH, 'final_df.csv'))
    final_df_not_qt = pd.read_csv(os.path.join(PREPROCESSED_PATH, 'final_df_not_qt.csv'))

    #   Shuffle
    final_df_qt = final_df_qt.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df_not_qt = final_df_not_qt.sample(frac=1, random_state=42).reset_index(drop=True)

    #   Save
    save_all_models(final_df_qt, final_df_not_qt, MODEL_SAVE_PATH)
    print('All models saved successfully...!!')