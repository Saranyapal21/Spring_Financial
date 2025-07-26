#   The below code implements the inference pipeline using the saved models (Refer to Models/)
#   Works for any models (tree or non-tree models except the Neural Network)

#   NOTE:-
#   While using tree-based models, always use 'final_df_not_qt' as the dataset
#   While using non-tree models, always use 'final_df_qt' as the dataset
#   The below example is inference using a tree-based ensemble model (The best model infact, with ~75% accuracy).
#   Due to this, 'final_df_not_qt' was used as the dataset

#   If you just want to perform inference using some datasets, run inference.py
#   Note that you dont need to pre-process data manually. It is automatically taken care of...
#   However, the format of the data must be maintained (should be similar to the current one)


import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from dataloader import pre_process_df




def perform_inference(final_df_qt, final_df_not_qt, MODEL_SAVE_PATH):
    # model_filename = "catboost_depth3_iter250_lr0.05.pkl"
    model_filename = "best_model.pkl"       #   Change it as needed
    model_path = os.path.join(MODEL_SAVE_PATH, model_filename)

    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(final_df_not_qt.drop('target', axis=1))
    try:
        y_prob = model.predict_proba(final_df_not_qt.drop('target', axis=1))[:, 1]
    except AttributeError:
        y_prob = None

    return {
        "predictions": y_pred,
        "probabilities": y_prob
    }






if __name__ == '__main__':
    env_path = Path(os.getcwd()) / 'Config' / '.env'
    load_dotenv(env_path)

    DATA_PATH = os.getenv('DATA_PATH')
    MODEL_SAVE_PATH = os.getenv('MODEL_PATH')

    df = pd.read_csv(DATA_PATH)
    final_df_qt, final_df_not_qt = pre_process_df(df, drop_high_corr=True)
    result = pd.DataFrame(perform_inference(final_df_qt, final_df_not_qt, MODEL_SAVE_PATH))

    print(result)