import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from dataloader import pre_process_df



#   Works for tree or non-tree models as well (All models except the Neural Network)
def perform_inference(final_df_qt, final_df_not_qt, MODEL_SAVE_PATH):
    model_filename = "catboost_depth3_iter250_lr0.05.pkl"
    model_path = os.path.join(MODEL_SAVE_PATH, model_filename)

    # Load the model from disk
    with open(model_path, "rb") as f:
        model = pickle.load(f)


    y_pred = model.predict(final_df_not_qt)
    try:
        y_prob = model.predict_proba(final_df_not_qt)[:, 1]
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