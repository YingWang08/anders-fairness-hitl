import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(path, exist_ok=True)

def save_results(results_dict, filename, results_dir='../results'):
    """
    将结果字典保存为CSV文件。
    results_dict: 包含模型名称和指标值的字典，例如 {'LR': {'R2':0.72, 'DI':0.58}, ...}
    filename: 保存的文件名（不含路径）
    results_dir: 结果目录，默认为'../results'
    """
    ensure_dir(results_dir)
    df = pd.DataFrame(results_dict).T  # 转置使模型名为行索引
    df.to_csv(os.path.join(results_dir, filename), index_label='Model')
    print(f"Results saved to {os.path.join(results_dir, filename)}")

def load_agriculture_data(data_dir='../data', split=True):
    """
    加载农业模拟数据集。
    返回: (X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test) 如果 split=True
          否则返回合并的DataFrame
    """
    train = pd.read_csv(os.path.join(data_dir, 'agriculture_train.csv'))
    val = pd.read_csv(os.path.join(data_dir, 'agriculture_val.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'agriculture_test.csv'))
    
    feature_cols = ['farm_area', 'labor', 'farming_years', 'irrigation', 'fertilizer', 'past_yield']
    target_col = 'resource_allocation'
    sensitive_col = 'region'
    
    # 编码敏感属性：A=0 (unprivileged), B/C=1 (privileged)
    def encode_region(r):
        return 0 if r == 'A' else 1
    
    for df in [train, val, test]:
        df['sensitive'] = df[sensitive_col].apply(encode_region)
    
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[target_col].values.astype(np.float32)
    s_train = train['sensitive'].values.astype(np.int32)
    
    X_val = val[feature_cols].values.astype(np.float32)
    y_val = val[target_col].values.astype(np.float32)
    s_val = val['sensitive'].values.astype(np.int32)
    
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test[target_col].values.astype(np.float32)
    s_test = test['sensitive'].values.astype(np.int32)
    
    if split:
        return X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test
    else:
        return train, val, test

def load_german_data(data_dir='../data', split=True, test_size=0.3, val_size=0.5, random_state=42):
    """
    加载UCI German Credit数据集。
    返回: (X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test) 如果 split=True
          否则返回原始DataFrame
    """
    file_path = os.path.join(data_dir, 'german_credit.csv')
    if not os.path.exists(file_path):
        # 如果文件不存在，自动下载
        print("German Credit dataset not found locally. Downloading from UCI...")
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
        df = pd.read_csv(url, header=None, delim_whitespace=True)
        ensure_dir(data_dir)
        df.to_csv(file_path, index=False, header=False)
    else:
        df = pd.read_csv(file_path, header=None)
    
    # 列名（根据UCI文档）
    columns = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
               'savings', 'employment', 'installment_rate', 'personal_status_sex', 'other_debtors',
               'present_residence', 'property', 'age', 'other_installment_plans', 'housing',
               'existing_credits', 'job', 'num_liable', 'telephone', 'foreign_worker', 'target']
    df.columns = columns
    
    # 目标变量：1=good, 2=bad -> 转换为1/0 (good=1, bad=0)
    df['target'] = (df['target'] == 1).astype(int)
    
    # 性别编码
    def gender_from_status(s):
        # A91: male (divorced/separated), A92: female (divorced/separated/married),
        # A93: male (single), A94: male (married/widowed), A95: female (single)
        if s in ['A91', 'A93', 'A94']:
            return 1  # male as privileged
        else:
            return 0  # female as unprivileged
    df['gender'] = df['personal_status_sex'].apply(gender_from_status)
    
    # 特征工程：将一些分类变量转换为虚拟变量
    # 为了简化，我们只使用部分数值特征，并添加几个重要分类变量的虚拟变量
    categorical_cols = ['checking_status', 'credit_history', 'purpose', 'savings', 'employment']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 所有数值列作为特征
    feature_cols = [c for c in df.columns if c not in ['target', 'personal_status_sex', 'gender']]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    s = df['gender'].values.astype(np.int32)
    
    if not split:
        return df
    
    # 划分训练/验证/测试集
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=test_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=val_size, random_state=random_state, stratify=y_temp)
    
    return X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test

def standardize_features(X_train, X_val=None, X_test=None):
    """
    对特征进行标准化（Z-score），返回标准化后的数据及scaler对象。
    如果只提供X_train，则仅拟合scaler并返回标准化后的X_train和scaler。
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    if X_val is not None and X_test is not None:
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    elif X_val is not None:
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        return X_train_scaled, X_val_scaled, scaler
    else:
        return X_train_scaled, scaler