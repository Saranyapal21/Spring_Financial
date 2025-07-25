import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from dataloader import pre_process_df
from train import save_all_models
from inference import perform_inference




if __name__ == '__main__':
    env_path = Path(os.getcwd()) / 'Config' / '.env'
    load_dotenv(env_path)

    DATA_PATH = os.getenv('DATA_PATH')
    MODEL_SAVE_PATH = os.getenv('MODEL_PATH')

    df = pd.read_csv(DATA_PATH)

    #   Pre-process data
    final_df_qt, final_df_not_qt = pre_process_df(df, drop_high_corr=True)

    #   Trains all (good) models on pre-processed data and saves them in Models/
    save_all_models(final_df_qt, final_df_not_qt, MODEL_SAVE_PATH)


    #   Perform inference
    result = pd.DataFrame(perform_inference(final_df_qt, final_df_not_qt, MODEL_SAVE_PATH))
    print(result)