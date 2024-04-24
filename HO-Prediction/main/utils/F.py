from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def ts_array_create(data_list, time_seq_len, pred_time, features, ffill_cols=[],two_hot_cols=[],merged_cols=[]):

    X_all = []
    Y_all_cls = []
    Y_all_fst = []
    files_record = []
    
    def vecot_to_num(v):
        num = 0.0
        for i, t in enumerate(v):
            if t != 0:
                num = i+t
                break
        return num

    def replace_zero_with_one(value):
        if value == 0:
            return 0
        else:
            return 1


    count = 0
    for file in tqdm(data_list):

        df = pd.read_csv(file)

        # Hard to change to a feature, delete it now.
        del df['Timestamp'], df['PCI'], df['EARFCN'], df['NR-PCI']

        # Two hot column
        for col in two_hot_cols:
            df[col] = df[col].apply(replace_zero_with_one)
            
        df[ffill_cols] = df[ffill_cols].replace(0, pd.NA)
        df[ffill_cols] = df[ffill_cols].ffill()
        for col in ffill_cols:
            if not pd.notna(df[col].iloc[0]):
                df = df[df[col].notna()]
        df.reset_index(drop=True, inplace=True)
        
        X = df[features]
        # Merged columns
        for cols in merged_cols:
            new_column = X[cols[:-1]].max(axis=1)
            col_num = X.columns.get_loc(cols[0])
            X = X.drop(cols[:-1], axis=1)
            X.insert(col_num, cols[-1], new_column)
        
        target = ['RLF_II', 'RLF_III']
        Y = df[target].copy()
        Y['RLF'] = Y.apply(lambda row: max(row['RLF_II'], row['RLF_III']), axis=1)
        Y.drop(columns=target, inplace=True)

        Xt_list = []
        Yt_list = []

        for i in range(time_seq_len):
            X_t = X.shift(periods=-i)
            X_t = X_t.to_numpy()
            Xt_list.append(X_t)

        Xt_list = np.stack(Xt_list, axis=0)
        Xt_list = np.transpose(Xt_list, (1,0,2))
        Xt_list = Xt_list[:-(time_seq_len + pred_time -1), :, :]

        for i in range(time_seq_len, time_seq_len+pred_time):
            Y_t = Y.shift(periods=-i)
            Y_t = Y_t.to_numpy()
            Yt_list.append(Y_t)

        Yt_list = np.stack(Yt_list, axis=0)
        Yt_list = np.transpose(Yt_list, (1,0,2))
        Yt_list = Yt_list[:-(time_seq_len + pred_time -1), :, :]
        Yt_list = np.squeeze(Yt_list)
        if pred_time == 1: 
            Yt_cls = np.where(Yt_list != 0, 1, 0) 
            Yt_fst = Yt_list
        else: 
            Yt_cls = np.where((Yt_list != 0).any(axis=1), 1, 0)
            Yt_fst = np.apply_along_axis(vecot_to_num, axis=1, arr=Yt_list)

        X_all.append(Xt_list)
        Y_all_cls.append(Yt_cls)
        Y_all_fst.append(Yt_fst)
        files_record.append((file, (count, count +len(Yt_cls))))
        count += len(Yt_cls)
        
    X_all = np.concatenate(X_all, axis=0)
    Y_all_cls = np.concatenate(Y_all_cls, axis=0)
    Y_all_fst = np.concatenate(Y_all_fst, axis=0)
    
    return X_all, Y_all_cls, Y_all_fst, files_record

# performance
def performance(model, dtest, y_test):
    
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    ACC = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred_proba)
    AUCPR = average_precision_score(y_test, y_pred_proba)
    P = precision_score(y_test, y_pred)
    R = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {ACC}; AUC: {AUC}; AUCPR: {AUCPR}; P: {P}; R: {R}; F1: {F1}")
    
    return ACC, AUC, AUCPR, P, R, F1

# Debug Function
def count_rlf(data_list):
    count = 0
    for f in data_list:
        df = pd.read_csv(f)
        for i in range(len(df)):
            if df['RLF_II'].iloc[i] or df['RLF_III'].iloc[i]:
                count += 1
    return count

def np_ary_to_df(arr, col_names):
    df = pd.DataFrame(arr, columns=col_names)
    return df
    
def find_original_input(ind, file_record, time_seq_len, ffill_cols):
    for (file, ind_range) in file_record:
        if ind_range[0]<=ind<ind_range[1]:
            target_file = file    
            tar_ind_range = ind_range
            
    df = pd.read_csv(target_file)
    df[ffill_cols] = df[ffill_cols].replace(0, pd.NA)
    df[ffill_cols] = df[ffill_cols].ffill()
    for col in ffill_cols:
        if not pd.notna(df[col].iloc[0]):
            df = df[df[col].notna()]
    df.reset_index(drop=True, inplace=True)
    return df[ind-tar_ind_range[0]:ind-tar_ind_range[0]+time_seq_len], target_file

def get_pred_result_ind(model, x, labels, X):
    TP, FP, TN, FN = [], [], [], [] 

    y_pred_proba = model.predict(x) 
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    for i, (pred, label, x) in enumerate(zip(y_pred, labels, X)):
        if pred != label:
            if label == 1: # FP analysis
                FP.append(i)
            else: # FN analysis
                FN.append(i)
        else: 
            if label == 1: # TP analysis
                TP.append(i)
            else:
                TN.append(i)        
    return TP, FP, TN, FN