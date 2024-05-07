#!/usr/bin/env python3
# coding: utf-8

# In[7]:


import pandas as pd
from map4 import MAP4Calculator
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm, tnrange
from glob import glob
import seaborn as sns


# A simple function to create a Morgan fingerprint as a numpy array

# In[2]:


def fp_as_array(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Build a dataframe from a SMILES file.
# 1. Read the SMILES
# 2. Calculate a MAP4 fingerprint and put that into the dataframe
# 3. Calculate a Morgan fingerprint and put thta into the dataframe

# In[3]:


def build_dataframe(input_file_name):
    df = pd.read_csv(input_file_name,header=None,sep=" ")
    df.columns = ["SMILES","Name","IC50"]
    m4_calc = MAP4Calculator(is_folded=True)
    df['mol'] = [Chem.MolFromSmiles(x) for x in df.SMILES]
    df['map4'] = [m4_calc.calculate(x) for x in df.mol]
    df['morgan'] = [fp_as_array(x) for x in df.mol]
    return df


# Benchmark a dataset.
# 1. Split into training and test sets using the sklearn defaults
# 2. Train and test an XGBoost regressor with MAP4 fingerprints and record the R<sup>2</sup>
# 3. Train and test an XGBoost regressor with Morgan fingerprints and record the R<sup>2</sup>
#
# Return lists of R<sup>2</sup> for MAP4 and Morgan

# In[41]:


def benchmark_dataset(df, name, cv_folds=5):
    map4_list = []
    morgan_list = []
    for i in tnrange(cv_folds, desc=name):
        train, test = train_test_split(df)

        # # XGB
        # map4_xgb = XGBRegressor()
        # X_train = np.array([x for x in train.map4.values])
        # y_train = train.IC50
        # map4_xgb.fit(X_train, y_train)
        # map4_pred = map4_xgb.predict(np.array([x for x in test.map4.values]))
        # map4_r2 = r2_score(test.IC50, map4_pred)
        # map4_list.append(map4_r2)

        # morgan_xgb = XGBRegressor()
        # morgan_xgb.fit(np.array([x for x in train.morgan.values]), train.IC50)
        # morgan_pred = morgan_xgb.predict(np.array([x for x in test.morgan.values]))
        # morgan_r2 = r2_score(test.IC50, morgan_pred)
        # morgan_list.append(morgan_r2)

        X_train = np.array([x for x in train.map4.values])
        y_train = train.IC50

        X_test = np.array([x for x in test.map4.values])
        y_test = test.IC50

        # RFR
        map4_rfr = RandomForestRegressor(n_estimators = 100,
                                         criterion = 'squared_error',
                                         #n_jobs = nprocs, # FBR: check if improves perfs
                                         oob_score = True,
                                         min_samples_leaf = 1,
                                         max_features = max_features, # FBR: fraction; could be optimized
                                         max_samples = None) # sklearn default
        map4_rfr.fit(X_train, y_train)
        map4_pred = map4_rfr.predict(X_test)
        map4_r2 = r2_score(y_test, map4_pred)
        map4_list.append(map4_r2)

        morgan_rfr = RandomForestRegressor(n_estimators = 100,
                                           criterion = 'squared_error',
                                           #n_jobs = nprocs, # FBR: check if improves perfs
                                           oob_score = True,
                                           min_samples_leaf = 1,
                                           max_features = max_features, # FBR: fraction; could be optimized
                                           max_samples = None) # sklearn default
        morgan_rfr.fit(X_train, y_train)
        morgan_pred = morgan_rfr.predict(X_test)
        morgan_r2 = r2_score(y_test, morgan_pred)
        morgan_list.append(morgan_r2)

    return (np.array(map4_list), np.array(morgan_list))


# Run the bechmark on a set of files.

# In[132]:


def run_benchmarks(file_spec):
    result_list = []
    for filename in glob(file_spec):
        df = build_dataframe(filename)
        name = filename.replace(".smi","")
        map4_res, morgan_res = benchmark_dataset(df,name)
        for map4,morgan in zip(map4_res,morgan_res):
            result_list.append([name,map4,morgan])
    return result_list


# Run the benchmark, on my MacBook Pro this took 21 minutes.

# In[ ]:


get_ipython().run_line_magic('time', 'result_list = run_benchmarks("*.smi")')


# Put the results into a dataframe and write the dataframe to disk.

# In[141]:


result_df = pd.DataFrame(result_list,columns=["dataset","map4","morgan"])
result_df.to_csv("map4_morgan_comparison.csv",index=False)
