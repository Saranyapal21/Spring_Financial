# Spring_Financial

This repository implements an ML pipeline for binary classification on any custom dataset. It was done as part of the take-home-assignment from Spring Financials. The overall steps involved in
the pipeline include the following:-

**Pipeline Flow:**
```
load → preprocess → train → save → batch inference
```

Inside the repository, you will find an organized structure of folders such as Data, Models, Notebooks and some python files such as `dataloader.py`, `train.py`, `inference.py` and `main.py`.
Inside the Notebooks/ folder, you will find two ipynb notebooks: `Modelling_experiments.ipynb` and `Preprocess_experiments.ipynb`. The names suggest the purpose of these notebooks. The former
was used to implement and compare different Pre-processing and Data transformation techniques and the later contains comparison of different modelling experiments. 

The python files are just a well-structured, proper version of only the steps that should be actually performed to achieve the best performance. All the experiemnts from the notebooks that did not provide
optimal results was removed in the python files. And the most important files (which form the actual pipeline) are:

**Key scripts:** `dataloader.py`, `train.py`, `inference.py`, `main.py`

---


## Steps to run the repo locally:-
I personally used the `uv package manager` to maintain this project. So the steps mentioned work well with `uv`. Follow below steps if you want to simply run the implementations locally:-
1. Clone the repo using:  
   ```bash
   git clone https://github.com/Saranyapal21/Spring_Financial.git
   cd Spring_Financial
   uv python install 3.12
   uv venv --python 3.12 .venv
2. Activate .venv (example for Unix-based systems):
   ```bash
   source .venv/bin/activate
   uv pip install --upgrade pip setuptools wheel
   rm -rf build dist *.egg-info
3. Install dependencies from pyproject.toml:
   ```bash
   uv pip install .

After this, you need to add a `Config` folder in the parent directory of the repository and create a `.env` file inside the Config folder. That `.env` file will contain paths to 
some of the important locations necessary to run the code properly. Please update the `.env` file using the steps mentioned below. The repo provides us the flexibility on training any model 
and running inference based on a custom dataset. Please refer below inorder to work with custom datasets.

---

## Pre-processing Custom Data and Model Training:-

**Before uploading any dataset, make sure the data follows the structure exactly as in the sample dataset, which is inside the Data/ folder (And not the one inside the Data/Pre_Processed).
Everything in your custom data should match the sample data, including the column names, column order etc.**

As mentioned earlier, this repository provides the flexibility to pre-process your own dataset (and not the one already uploaded), train model on it and then perform inference. The below steps 
shows it how:
- Create a `Config` folder and in the parent directory of the repository and create a `.env` file inside it
- Store these important paths inside the `.env` file:
  ```Bash
   DATA_PATH=
   MODEL_SAVE_PATH=
   PRE_PROCESSED_PATH=

The main purpose of these different environement variables mentioned above are as follows:- 
- `DATA_PATH` is the path to the actual unprocessed data. After you upload your own data inside the `Data/` folder, copy its path and update the `DATA_PATH` inside the .env with that path.
- `PRE_PROCESSED_PATH` is the path to the folder that decides where your pre-processed data will be stored. Just use the `Data/Pre_Processed` as the path for the `PRE_PROCESSED_PATH` env variable and then modify the codes accordingly as per your filename.
-  `MODEL_SAVE_PATH` refers to the path to the folder where your model (trained on the pre-processed data) will be saved after training. Just use the `Models/` as the path for the `MODEL_SAVE_PATH` env variable and then modify the codes accordingly as per your filename.

**Important:-**
- Using the above simple techniques, you can easily train a model on the custom data. Pre-processing is automatically taken care of directly. It is not required to first pre-process and then
train the model using that pre-processed dataset. If your paths are correct and .env is set-up properly, then for training model, simply running `train.py` is sufficient.
- Below is a summarisation of how to run the entire pipeline or just parts of the pipeline:
  - To just pre-process the custom data, run `dataloader.py` using the command `python3 dataloader.py` in the correct path. Your pre-processed data will be saved inside `Data/Pre_Processed/` automatically
  - To just train a model on the custom data, run `train.py` using the command `python3 train.py` in the correct path. Pre-processing is automatically taken care of as mentioned earlier
  - Suppose you want to run inference on your custom dataset. Just run the command `inference.py`. your custom data is pre-processed automatically and inference is performed using the saved
  models in .pkl format

The `main.py` binds all these 3 important modules of the pipeline together, providing a complete end-to-end pipeline ranging from pre-processing to training and finally inference; all using a
single command. Just use `python3 main.py` to run the end-to-end pipeline.


---

### Purpose of current saved .pkl models

The problem statement mentioned training ML models on the **entire** dataset provided. In accordance to that, all the top-performing models are saved inside the Models/ folder as .pkl file and are trained on the entire dataset.
Only the Neural Network model is stored as .pt (instead of .pkl) where we need to load state_dict() in order to perform inference with that model. As our experiments suggest, the **CatBoost model**, named as
`catboost_depth3_iter250_lr0.05.pkl` is the best performing model. Even the current inference.py uses that model to perform inference. Of course, you can customise that and use a different 
model as per your need and convenience.


So, if you simply want to perform inference using the best model, upload your dataset, follow the setting up steps for the repo as mentioned above and run inference.py. The predictions 
(predicted lables as well as the predicted_probabilities) will be printed in the terminal for each of the datapoint present in the inference


---

### Brief overview of techniques used during loading data and training:-

**dataloader.py**

   - **drop_cols_and_fill_na_values**: Drops noisy column and fills missing values using **median**.
   - **create_cat_num_and_oth_features**: Splits columns into numerical, categorical, and other types (3 types total)
   - **transform_num_and_oth_features**: Applies **quantile transformation** to selected numerical features.
   - **standardize_rem_features**: Standard scales a few specific numerical columns.
   - **transform_cat_features**: One-hot encodes categorical variables with different strategies based on unique value count.
   - **merge_and_concatenate**: Combines all processed numerical, categorical, and untouched columns into final dataframe.
   - **pre_process_df**: Master function to handle full preprocessing pipeline. Returns two versions of processed data.


**train.py**

   - **model_fit_and_save**: Trains and saves classical ML models (SVC, LogisticRegression) with optional polynomial features.
   - **save_simple_models**: Trains and saves baseline classical models like LogisticRegression and SVMs.
   - **train_and_save_tree_model**: Trains and saves a tree-based model (CatBoost, XGBoost, etc.).
   - **save_top_tree_models**: Trains and saves all top tree-based ensemble models.
   - **Network_v1**: Simple feedforward neural network for binary classification.
   - **train_and_save_neural_net_model**: Trains the neural network and saves the model weights.
   - **save_all_models**: Trains and saves all best-performing models using both versions of the preprocessed data.
