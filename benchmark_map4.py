#!/usr/bin/env python3
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed
from map4 import MAP4Calculator
from atom_pairs import encode_molecules_unfolded as ap_encode
from uhd import encode_molecules_unfolded as uhd_encode
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm, tnrange
from glob import glob

# A simple function to create a Morgan fingerprint as a numpy array

# In[2]:


def fp_as_array(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr # FBR: numpy.ndarray of int64 w/ length 2048


# Build a dataframe from a SMILES file.
# 1. Read the SMILES
# 2. Calculate a MAP4 fingerprint and put that into the dataframe
# 3. Calculate a Morgan fingerprint and put thta into the dataframe

# In[3]:


def build_dataframe(input_file_name):
    df = pd.read_csv(input_file_name, header=None, sep=" ")
    df.columns = ["SMILES", "Name", "IC50"]
    m4_calc = MAP4Calculator(is_folded=True)
    df['mol'] = [Chem.MolFromSmiles(x) for x in df.SMILES]
    df['map4'] = [m4_calc.calculate(x) for x in df.mol]
    df['morgan'] = [fp_as_array(x) for x in df.mol]
    # FBR: folded AP might benefit from a significantly larger size: also try 4096, 8192 and 16384
    # for unfolded version: AP has 14086 features on erbB1; highest dimensional dataset
    df['ap'] = ap_encode(df.mol)
    df['uhd'] = uhd_encode(df.mol)
    return df

# Benchmark a dataset.
# 1. Split into training and test sets using the sklearn defaults
# 2. Train and test an XGBoost regressor with MAP4 fingerprints and record the R<sup>2</sup>
# 3. Train and test an XGBoost regressor with Morgan fingerprints and record the R<sup>2</sup>
#
# Return lists of R<sup>2</sup> for MAP4 and Morgan

# In[15]:


# FBR: I lowered from 10xCV to 5xCV for faster bench
def benchmark_dataset(df, name, mtry, cv_folds=5):
    map4_list = []
    morgan_list = []
    ap_list = []
    uhd_list = []
    for i in tnrange(cv_folds, desc=name):
        train, test = train_test_split(df)

        y_train = train.IC50
        y_test = test.IC50

        nprocs = 24 # FBR: for my workstation

        # # XGB -----------------------------------------------------------------
        # # MAP4
        # X_train = np.array([x for x in train.map4.values])
        # X_test = np.array([x for x in test.map4.values])
        # map4_xgb = XGBRegressor()
        # map4_xgb.fit(X_train, y_train)
        # map4_pred = map4_xgb.predict(X_test)
        # map4_r2 = r2_score(y_test, map4_pred)
        # map4_list.append(map4_r2)
        # # ECFP
        # X_train = np.array([x for x in train.morgan.values])
        # X_test = np.array([x for x in test.morgan.values])
        # morgan_xgb = XGBRegressor()
        # morgan_xgb.fit(X_train, y_train)
        # morgan_pred = morgan_xgb.predict(X_test)
        # morgan_r2 = r2_score(y_test, morgan_pred)
        # morgan_list.append(morgan_r2)
        # # AP
        # X_train = np.array([x for x in train.ap.values])
        # X_test = np.array([x for x in test.ap.values])
        # ap_xgb = XGBRegressor()
        # ap_xgb.fit(X_train, y_train)
        # ap_pred = ap_xgb.predict(X_test)
        # ap_r2 = r2_score(y_test, ap_pred)
        # ap_list.append(ap_r2)

        # RFR -----------------------------------------------------------------
        # MAP4
        X_train = np.array([x for x in train.map4.values])
        X_test = np.array([x for x in test.map4.values])
        map4_rfr = RandomForestRegressor(n_estimators = 50,
                                         criterion = 'squared_error',
                                         n_jobs = nprocs,
                                         oob_score = True,
                                         min_samples_leaf = 1,
                                         max_features = mtry,
                                         max_samples = None) # sklearn default
        map4_rfr.fit(X_train, y_train)
        map4_pred = map4_rfr.predict(X_test)
        map4_r2 = r2_score(y_test, map4_pred)
        map4_list.append(map4_r2)
        # ECFP
        X_train = np.array([x for x in train.morgan.values])
        X_test = np.array([x for x in test.morgan.values])
        morgan_rfr = RandomForestRegressor(n_estimators = 50,
                                           criterion = 'squared_error',
                                           n_jobs = nprocs,
                                           oob_score = True,
                                           min_samples_leaf = 1,
                                           max_features = mtry,
                                           max_samples = None) # sklearn default
        morgan_rfr.fit(X_train, y_train)
        morgan_pred = morgan_rfr.predict(X_test)
        morgan_r2 = r2_score(y_test, morgan_pred)
        morgan_list.append(morgan_r2)
        # AP
        X_train = np.array([x for x in train.ap.values])
        X_test = np.array([x for x in test.ap.values])
        ap_rfr = RandomForestRegressor(n_estimators = 50,
                                       criterion = 'squared_error',
                                       n_jobs = nprocs,
                                       oob_score = True,
                                       min_samples_leaf = 1,
                                       max_features = mtry,
                                       max_samples = None) # sklearn default
        ap_rfr.fit(X_train, y_train)
        ap_pred = ap_rfr.predict(X_test)
        ap_r2 = r2_score(y_test, ap_pred)
        ap_list.append(ap_r2)
        # UHD
        X_train = np.array([x for x in train.uhd.values])
        X_test = np.array([x for x in test.uhd.values])
        uhd_rfr = RandomForestRegressor(n_estimators = 50,
                                       criterion = 'squared_error',
                                       n_jobs = nprocs,
                                       oob_score = True,
                                       min_samples_leaf = 1,
                                       max_features = mtry,
                                       max_samples = None) # sklearn default
        uhd_rfr.fit(X_train, y_train)
        uhd_pred = uhd_rfr.predict(X_test)
        uhd_r2 = r2_score(y_test, uhd_pred)
        uhd_list.append(uhd_r2)

    return (np.array(map4_list), np.array(morgan_list), np.array(ap_list), np.array(uhd_list))


# Run the bechmark on a set of files.

# In[16]:


def run_benchmarks(mtry):
    result_list = []
    for filename in glob('*.smi'):
        df = build_dataframe(filename)
        name = filename.replace(".smi", "")
        map4_res, morgan_res, ap_res, uhd_res = benchmark_dataset(df, name, mtry)
        for map4, morgan, ap, uhd in zip(map4_res, morgan_res, ap_res, uhd_res):
            result_list.append([name, map4, morgan, ap, uhd])
    return (mtry, result_list)


# Run the benchmark, on my MacBook Pro this took 21 minutes.
#mtrys = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
mtrys = [0.1]
# get_ipython().run_line_magic('time', 'result_list = run_benchmarks("*.smi")')
mtry_result_lists = Parallel(n_jobs=1)(delayed(run_benchmarks)(mtry) for mtry in mtrys)
# Put the results into a dataframe and write the dataframe to disk.
for mtry, result_list in mtry_result_lists:
    result_df = pd.DataFrame(result_list, columns=["dataset", "map4", "morgan", "ap", "uhd"])
    result_df.to_csv("map4_morgan_comparison_%f.csv" % mtry, index=False)
