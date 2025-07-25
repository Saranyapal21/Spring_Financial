import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import QuantileTransformer, StandardScaler



def drop_cols_and_fill_na_values(df):           
    df_copy = df.copy()
    df_copy.drop("feature_20", axis=1, inplace=True)    #   Drop this feature
    features = ['feature_2', 'feature_12', 'feature_15', 'feature_17']  #   Only these features have missing values

    for feature in features:
        median_val = df_copy[feature].median()
        df_copy[feature] = df_copy[feature].fillna(median_val)

    return df_copy



def create_cat_num_and_oth_features(df):  
    num_features, cat_feature, oth_features = [], [], []

    for col in df.columns:
        if df[col].value_counts().shape[0] >= 1500:
            num_features.append( col )
        elif df[col].value_counts().shape[0] <= 100:
            cat_feature.append( col )
        else:
            oth_features.append( col )

    cat_features = [col for col in cat_feature if col != 'target']   #  target column removed...!!

    return num_features, cat_features, oth_features



#   Will return a new_df having the transformed features
def transform_num_and_oth_features(df_copy):
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    
    qt_features = ['feature_2', 'feature_12', 'feature_15', 'feature_17', 'feature_6', 'feature_7', 'feature_9', 'feature_38', 'feature_39', 'feature_42', 'feature_43', 'feature_44', 'feature_46', 'feature_0', 'feature_1', 'feature_3', 'feature_4', 'feature_10', 'feature_21', 'feature_33', 'feature_34', 'feature_40', 'feature_41', 'feature_45', 'feature_47']
    qt_not_transformed = df_copy[qt_features]

    qt_transformed = pd.DataFrame(qt.fit_transform(df_copy[qt_features]),
                              columns=[f for f in qt_features])


    return qt_transformed, qt_not_transformed



#   Standardize mentioned ones or all of the features
def standardize_rem_features(df):
    unmodified_features = ['feature_11', 'feature_22', 'feature_48']

    scaler = StandardScaler()
    standardized = scaler.fit_transform(df[unmodified_features])
    
    standardized_df = pd.DataFrame(standardized, columns=unmodified_features, index=df.index)
    
    return standardized_df



#   Will return a new_df having the one-hot encoded features
def transform_cat_features(df_copy, cat_features):
    encoded_parts = []

    for col in cat_features:
        unique_vals = df_copy[col].nunique()
        val_counts = df_copy[col].value_counts()

        if unique_vals == 2:
            encoded_parts.append(df_copy[[col]])
        elif unique_vals <= 5:
            ohe_df = pd.get_dummies(df_copy[col], prefix=col, drop_first=True).astype(int)
            encoded_parts.append(ohe_df)
        else:
            top_classes = val_counts.index[:3]  # Top 3
            new_cols = pd.DataFrame()

            for cls in top_classes:
                new_cols[f"{col}_{cls}"] = (df_copy[col] == cls).astype(int)

            new_cols[f"{col}_Other"] = (~df_copy[col].isin(top_classes)).astype(int)
            encoded_parts.append(new_cols)

    cat_encoded_df = pd.concat(encoded_parts, axis=1)
    return cat_encoded_df




#   This will merge all the 3 cases: transformed numerical, trasformed categorical and untransformed features
#   It will then return a new_df which is perfect for modelling
def merge_and_concatenate(num_and_oth_df, cat_df, original_df):
    final_df = pd.concat([cat_df, standardize_rem_features(original_df), num_and_oth_df, original_df['target']], axis=1)
    return final_df



#   This single function does all the pre-processing
def pre_process_df(original_df, drop_high_corr=True):
    num_features, cat_features, oth_features = create_cat_num_and_oth_features(original_df)
    df_no_missing = drop_cols_and_fill_na_values(original_df)    #   No categorical cols will be affected


    #   For fitting tree-based models like XgBoost etc, data transfomation especially 
    #   Quantile-transform and StandardScaler are often known to detoriate performance.
    #   So, while fitting those tree-based models, we must not perform this step

    qt_df, qt_not_df = transform_num_and_oth_features(df_no_missing)
    cat_df = transform_cat_features(df_no_missing, cat_features)


    final_df = merge_and_concatenate(qt_df, cat_df, df_no_missing)
    final_df_not_qt = merge_and_concatenate(qt_not_df, cat_df, df_no_missing)


    if drop_high_corr == True:
        final_df.drop(['feature_38', 'feature_45'], axis=1, inplace=True)
        final_df_not_qt.drop(['feature_38', 'feature_45'], axis=1, inplace=True)
    

    return final_df, final_df_not_qt




if __name__ == '__main__':

    env_path = Path(os.getcwd()) / 'Config' / '.env'
    load_dotenv(env_path)

    DATA_PATH = os.getenv('DATA_PATH')
    PREPROCESSED_PATH = os.getenv('PREPROCESSED_PATH')
    df = pd.read_csv(DATA_PATH)

    final_df, final_df_not_qt = pre_process_df(df, drop_high_corr=True)    #   'False' if you don't want to drop high corr features

    print(final_df.shape)
    print(final_df_not_qt.shape)
    
    file_path_qt = os.path.join(PREPROCESSED_PATH, 'final_df.csv')
    file_path_not_qt = os.path.join(PREPROCESSED_PATH, 'final_df_not_qt.csv')

    final_df.to_csv(file_path_qt, index=False)
    final_df_not_qt.to_csv(file_path_not_qt, index=False)