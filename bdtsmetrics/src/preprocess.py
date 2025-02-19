import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
from tslearn.datasets import UCR_UEA_datasets
import pickle
import mgzip
from .utils import show_with_start_divider, show_with_end_divider, make_sure_path_exist, MinMaxScaler
import json


# adapt from https://github.com/TheDatumOrg/VUS
def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    base = 3
    nobs = len(data)
    nlags = int(min(10 * np.log10(nobs), nobs - 1))
    auto_corr = acf(data, nlags=nlags, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
    
    
def sliding_window_view(data, window_size, step=1):
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")
    L, C = data.shape  # Length and Channels
    if L < window_size:
        raise ValueError("Window size must be less than or equal to the length of the array")

    # Calculate the number of windows B
    B = L - window_size + 1
    
    # Shape of the output array
    new_shape = (B, window_size, C)
    
    # Calculate strides
    original_strides = data.strides
    new_strides = (original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)

    # Create the sliding window view
    strided_array = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    return strided_array

def preprocess_data(cfg):
    show_with_start_divider(f"Data preprocessing with settings:{cfg}")

    # Parse configs
    ori_data_path = cfg.get('original_data_path',None)
    output_ori_path = cfg.get('output_ori_path',r'./data/ori/')
    dataset_name = cfg.get('dataset_name','dataset')
    use_ucr_uea_dataset = cfg.get('use_ucr_uea_dataset',None)
    ucr_uea_dataset_name = cfg.get('ucr_uea_dataset_name',None)
    seq_length = cfg.get('seq_length',None)
    valid_ratio = cfg.get('valid_ratio',0.1)
    do_normalization = cfg.get('do_normalization',True)

    # Read original data
    if not os.path.exists(ori_data_path):
        show_with_end_divider(f'Original file path {ori_data_path} does not exist.')
        return None
    _, ext = os.path.splitext(ori_data_path)
    try:
        if use_ucr_uea_dataset:
            X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(ucr_uea_dataset_name)
            X_train = X_train.reshape(X_train.shape[1], X_train.shape[0])
            ori_data = X_train.copy()
            
            for i in range(ori_data.shape[1]):
                ori_data[:,i] = pd.Series(ori_data[:,i]).interpolate().values

        if ext in ['.csv']:
            ori_data = np.loadtxt(ori_data_path, delimiter = ",", skiprows = 1)
        elif ext in ['.pkl']:
            try:
                with mgzip.open(ori_data_path, 'rb') as f:
                    ori_data = pickle.load(f)
            except (OSError, IOError):
                # If mgzip fails, try reading it as a regular pickle file
                with open(ori_data_path, 'rb') as f:
                    ori_data = pickle.load(f)
        else:
            show_with_end_divider(f"Error: Unsupported file extension: {ext}")
            return None
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None
    
    # Check and interpolate missing values
    if np.isnan(ori_data).any():
        if not isinstance(ori_data, pd.DataFrame):
            df = pd.DataFrame(ori_data)
            df = df.interpolate(axis=1)
        else:
            df = ori_data.interpolate(axis=1)
        ori_data = df.to_numpy()

    # Determine the data length
    if seq_length:
        if seq_length>0 and seq_length<=ori_data.shape[0]:
            seq_length = int(seq_length)
        else:
            window_all = []
            for i in range(ori_data.shape[1]):
                window_all.append(find_length(ori_data[:,i]))

            seq_length = int(np.mean(np.array(window_all)))
    
    # Slice the data by sliding window
    # windowed_data = np.lib.stride_tricks.sliding_window_view(ori_data, window_shape=(seq_length, ori_data.shape[1]))
    # windowed_data = np.squeeze(windowed_data, axis=1)
    windowed_data = sliding_window_view(ori_data, seq_length)
    
    # Shuffle
    idx = np.random.permutation(len(windowed_data))
    data = windowed_data[idx]
    print('Data shape:', data.shape) 

    train_len = int(data.shape[0] * (1 - valid_ratio))
    train_data = data[:train_len]
    valid_data = data[train_len:]

    if do_normalization:
        scaler = MinMaxScaler()        
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)
    
    # Save preprocessed data
    output_path = os.path.join(output_ori_path,dataset_name)
    make_sure_path_exist(output_path+os.sep)
    with mgzip.open(os.path.join(output_path,f'{dataset_name}_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with mgzip.open(os.path.join(output_path,f'{dataset_name}_valid.pkl'), 'wb') as f:
        pickle.dump(valid_data, f)

    show_with_end_divider(f'Preprocessing done. Preprocessed files saved to {output_path}.')
    return train_data, valid_data

def load_preprocessed_data(cfg):
    show_with_start_divider(f"Load preprocessed data with settings:{cfg}")

    # Parse configs
    dataset_name = cfg.get('dataset_name','dataset')
    output_ori_path = cfg.get('output_ori_path',r'./data/ori/')

    file_path = os.path.join(output_ori_path,dataset_name)
    train_data_path = os.path.join(file_path,f'{dataset_name}_train.pkl')
    valid_data_path = os.path.join(file_path,f'{dataset_name}_valid.pkl')

    # Read preprocessed data
    if not os.path.exists(train_data_path) or not os.path.exists(valid_data_path):
        show_with_end_divider(f'Error: Preprocessed file in {file_path} does not exist.')
        return None
    try:
        with mgzip.open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        with mgzip.open(valid_data_path, 'rb') as f:
            valid_data = pickle.load(f)
    except Exception as e:
        show_with_end_divider(f"Error: An error occurred during reading data: {e}.")
        return None

    show_with_end_divider(f'Preprocessed dataset {dataset_name} loaded.')
    return train_data, valid_data


def bd_sliding_window_view(data, window_size, step=1):
    if data.ndim != 2:
        raise ValueError("Input array must be 2D")
    L, C = data.shape  # Length and Channels
    if L < window_size:
        raise ValueError("Window size must be less than or equal to the length of the array")

    # Calculate the number of windows B
    B = L // window_size
    # B = L - window_size + 1

    # Shape of the output array
    new_shape = (B, window_size, C)

    # Calculate strides
    original_strides = data.strides
    new_strides = (window_size * original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)
    # new_strides = (original_strides[0],) + original_strides  # (stride for L, stride for W, stride for C)

    # Create the sliding window view
    strided_array = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    # strided_array = np.transpose(strided_array, axes=(0, 2, 1)) #(b c l)
    return strided_array

def normalize(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm


def load_my_own_data_numpy(path):
    data = np.load(path)
    norm_data = normalize(data)
    return norm_data

def load_my_own_data_csv(path, seq_len):
    data = pd.read_csv(path)
    data = data.to_numpy()
    if len(data.shape) != 2:
        data = data.reshape((df_real.shape[0],1))
        
    data = bd_sliding_window_view(data, seq_len)
    norm_data = normalize(data)
    return norm_data

def load_my_own_data_csv_v2(path, seq_len):
    data = pd.read_csv(path)
    data = data.to_numpy()
    if len(data.shape) != 2:
        data = data.reshape((df_real.shape[0],1))
        
    data = sliding_window_view(data, seq_len)
    norm_data = normalize(data)
    return norm_data
    

def load_from_df(df, seq_len):
    data = df.to_numpy()
    if len(data.shape) != 2:
        data = data.reshape((df_real.shape[0], 1))

    data = bd_sliding_window_view(data, seq_len)
    norm_data = normalize(data)
    return norm_data

def extract_ts_from_csv(path, seq_len, non_ts_cols):
    df = pd.read_csv(path)
    num_non_ts_cols = len(non_ts_cols)
    # extract ts data and convert to the shape required by next step
    print("number of non ts columns:", num_non_ts_cols)
    ts = df[df.columns[num_non_ts_cols:]]
    ts = ts.to_numpy()
    # print(extracted_ts_col_names)
    print("temporal data shape: {}, seq_len is: {}".format(ts.shape, seq_len))
    if ts.shape[1] % seq_len != 0:
        raise Exception("length of time series data must be divisble by seq_len")
    ts = np.reshape(ts, (ts.shape[0], seq_len, ts.shape[1] // seq_len))
    dim = ts.shape[0] * ts.shape[1]
    dim2 = ts.shape[2]
    ts = np.reshape(ts, (dim, dim2))
    # convert ts data back to df
    ts_df = pd.DataFrame(ts, columns=extracted_ts_col_names)

    return ts_df
    

            
    

