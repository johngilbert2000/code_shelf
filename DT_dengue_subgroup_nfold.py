
# Import Modules

import datetime
from pathlib import Path
import os
import argparse

import unittest

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp

# My Own Modules
from pyfiles.autometrics import show_metrics, subgroup_analysis, conf_and_auc
import pyfiles.autometrics
# import pyfiles.df_converter

# Data Information

# filename = 'patient_year_vital_lab_exam_add-on_death_outcome_comorbidity_TCIC_dengue_suspected_bmi_ER_label_missing_mask.csv'
filename = 'patients_cleaned.csv'
path = Path('../mydata') # folder where csv is accessed
save_path = Path('../mydata/myoutput/temp') # folder where results will be stored

# move up one directory

# original_dir = os.getcwd()
# new_dir = original_dir/Path('..')
# os.chdir(new_dir)


# PARAMETERS

# obtain parameters from command line
parser = argparse.ArgumentParser("simple_example")
parser.add_argument("-splits", default=5, help="Number of folds used for cross validation (default: 5).", type=int)
parser.add_argument("-recalls", default=[0.85,0.90,0.95], nargs='+', help="Sensitivities used for calculating results (default: [0.85,0.90,0.95]).", type=float)
parser.add_argument("-tag", default='', help="String added to the end of each filename in results (default: '').", type=str)
args = parser.parse_args()

# if not args:
#     splits = 10 # number of folds for cross validation
#     recalls = [0.85,0.90,0.95] # sensitivities used for calculating results
#     tag = '' # added to the end of each filename in results
# else:
splits = args.splits
recalls = [i for i in args.recalls]
tag = args.tag

# Imported columns from CSV
desired_cols = ['age','sex','Temp','exam_WBC','exam_Plt', 'Opd_Visit_Date',
                'ER', 'Heart Disease', 'CVA', 'CKD', 'Severe Liver Disease', 
                'DM', 'Hypertension', 'Cancer without Metastasis', 'Cancer with Metastasis',
                'lab_result']

# Features used for training + dependent variable
train_cols = ['age','Temp','exam_WBC','exam_Plt','lab_result']


# Features used for creating validation subgroups (includes features from train_cols)
subgroup_cols = ['age','sex','Temp','exam_WBC','exam_Plt', 'week',
                'ER', 'Heart Disease', 'CVA', 'CKD', 'Severe Liver Disease', 
                'DM', 'Hypertension', 'Cancer without Metastasis', 'Cancer with Metastasis',
                'lab_result']

# Columns to be dropped after creating validation subgroups
drop_cols = list(set(subgroup_cols) - set(train_cols))

dep_var = 'lab_result'

# CONVERT CSV TO DF

# Read CSV with desired columns
# desired_cols defined in Parameters section
df = pd.read_csv(path/filename, usecols=desired_cols)

# Randomize data
df = df.iloc[np.random.permutation(len(df))]

# Convert Opd_Visit_Date to week of year format
if 'week' not in df.columns:
    week_numbers = [datetime.datetime.strptime(d, "%Y/%m/%d").isocalendar()[1] for d in df['Opd_Visit_Date']]
    df.insert(0, 'week', week_numbers)
if 'Opd_Visit_Date' in df.columns:
    df.drop(columns=['Opd_Visit_Date'],inplace=True)

# Convert 男 and 女 to 0 and 1 in column 'sex'
df_male_indx = df[df['sex']=='男']
df_female_indx = df[df['sex']=='女']

for i in df_male_indx.index.tolist():
    df.at[int(i),'sex'] = 0
for i in df_female_indx.index.tolist():
    df.at[int(i),'sex'] = 1

    
# SEPARATE DF INTO SUBSETS 

dataframe = df

# Equal length subsets of original dataframe
len_df = len(dataframe)
cut_indices = [int(i*(1/splits)*len_df) for i in range(0,splits+1)]
cut_indices = zip(cut_indices[:-1], cut_indices[1:])
subsets = [dataframe[i:j] for i,j in cut_indices]

# Build Training Set from Subsets
trains = [pd.concat(subsets[1:], axis=0)]
for n in range(1,splits):
    trains += [pd.concat(subsets[:n]+subsets[n+1:], axis=0)] 

    
# Separate Dataframe into Subgroups

dfs = [] # stores dataframes

for modelnum in range(1,splits+1):

    dataframe = subsets[modelnum-1] # validation set

    # age
    df_age_under_18 = dataframe[dataframe['age']<18]
    df_age_18_to_65 = dataframe[(dataframe['age']>=18) & (dataframe['age']<65)]
    df_age_over_eq_65 = dataframe[dataframe['age']>=65]

    # sex
    df_female = dataframe[dataframe['sex']==1]
    df_male = dataframe[dataframe['sex']==0]

    # week
    df_wks_35 = dataframe[dataframe['week']<=35]
    df_wks_35_to_40 = dataframe[(dataframe['week']>35) & (dataframe['week']<=40)]
    df_wks_over_40 = dataframe[dataframe['week']>40]

    # Temp
    df_temp_over_38 = dataframe[dataframe['Temp']>38]
    df_temp_under_eq_38 = dataframe[dataframe['Temp']<=38]

    # exam_WBC
    df_wbc_low = dataframe[dataframe['exam_WBC']<3.2]
    df_wbc_normal = dataframe[(dataframe['exam_WBC']>=3.2) & (dataframe['exam_WBC']<10)]
    df_wbc_high = dataframe[dataframe['exam_WBC']>10]

    # exam_Plt
    df_plt_low = dataframe[dataframe['exam_Plt']<100]
    df_plt_high = dataframe[dataframe['exam_Plt']>=100]

    # Comorbidities
    df_heart_disease = dataframe[dataframe['Heart Disease']==True]
    df_cva = dataframe[dataframe['CVA']==True]
    df_ckd = dataframe[dataframe['CKD']==True]
    df_liver = dataframe[dataframe['Severe Liver Disease']==True]
    df_dm = dataframe[dataframe['DM']==True]
    df_hypertension = dataframe[dataframe['Hypertension']==True]


    df_cancer1 = dataframe[(dataframe['Cancer with Metastasis']==True)]
    df_cancer2 = dataframe[(dataframe['Cancer without Metastasis']==True)]
    df_cancer = pd.concat([df_cancer1, df_cancer2], axis=0)

    df_er = dataframe[dataframe['ER']==True]

    frame = [df_age_under_18, df_age_18_to_65, df_age_over_eq_65, df_female, df_male, df_wks_35, df_wks_35_to_40, 
          df_wks_over_40, df_temp_over_38, df_temp_under_eq_38, df_wbc_low, df_wbc_normal, df_wbc_high, 
          df_plt_low, df_plt_high, df_heart_disease, df_cva, df_ckd, df_liver, df_dm, df_hypertension, 
          df_cancer, df_er]

    dfs_names = ['df_age_under_18', 'df_age_18_to_65', 'df_age_over_eq_65', 'df_female', 'df_male', 'df_wks_35', 'df_wks_35_to_40', 
          'df_wks_over_40', 'df_temp_over_38', 'df_temp_under_eq_38', 'df_wbc_low', 'df_wbc_normal', 'df_wbc_high', 
          'df_plt_low', 'df_plt_high', 'df_heart_disease', 'df_cva', 'df_ckd', 'df_liver', 'df_dm', 'df_hypertension', 
          'df_cancer', 'df_er']

    dfs += [frame]

# REMOVE UNUSED FEATURES
    
# drop_cols defined in Parameters section

for model_indx in range(len(dfs)):
    dataframes = dfs[model_indx]
    train_df = trains[model_indx]
    valid_df = subsets[model_indx]
    
    # Remove columns of unused features in validation subgroups
    for i in range(len(dataframes)):
        if drop_cols[0] in dataframes[i].columns:
            dataframes[i].drop(columns=drop_cols,inplace=True)

    # Remove columns of unused features in training dataset
    if drop_cols[0] in train_df.columns:
        train_df.drop(columns=drop_cols,inplace=True)

    # Remove columns of unused features in full validation dataset
    if drop_cols[0] in valid_df.columns:
        valid_df.drop(columns=drop_cols,inplace=True)
        

# REMOVE NANS

for model_indx in range(len(dfs)):
    dataframes = dfs[model_indx]
    train_df = trains[model_indx]
    valid_df = subsets[model_indx]

    # Option A: Drop NaNs

    # Remove NaN values in Validation Subgroups
    for i in range(len(dataframes)):
        dataframes[i].dropna(how='any',inplace=True)

    # Remove NaN values in Training and Validation Sets
    train_df.dropna(how='any',inplace=True)
    valid_df.dropna(how='any',inplace=True)


    # Option B: Convert NaNs to median values

    # for i in range(len(dataframes)):
    #     dataframes[i] = NaN_converter(dataframes[i])

    # train_df = NaN_converter(train_df)
    # valid_df = NaN_converter(valid_df)
    

# TRAINING
model_classifiers = []
model_preds = []
model_scores = []
model_valid_Ys = []
imgs = []
for model_indx in range(len(dfs)):
    train_df = trains[model_indx]
    valid_df = subsets[model_indx]

    # MORE DATA PREPARATION
    
    # Separate inputs (X) from outputs (Y)
    train_X = train_df.loc[:,train_df.columns != 'lab_result']
    train_Y = train_df.loc[:,train_df.columns == 'lab_result']
    valid_X = valid_df.loc[:,valid_df.columns != 'lab_result']
    valid_Y = valid_df.loc[:,valid_df.columns == 'lab_result']

    # Convert to numpy arrays
    train_X = train_X.to_numpy('float64')
    train_Y = train_Y.to_numpy('float64').flatten()
    valid_X = valid_X.to_numpy('float64')
    valid_Y = valid_Y.to_numpy('float64').flatten()
    
    model_valid_Ys += [valid_Y] # save for later
    
    # TRAIN

    clf = DecisionTreeRegressor(max_depth=5)
    
    clf.fit(train_X, train_Y)
    Y_pred = clf.predict(valid_X)
    score = 1 - metrics.mean_squared_error(valid_Y, Y_pred)
    
    # create image
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, class_names=[dep_var], feature_names=train_cols[:-1],
                   max_depth=None)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    img = Image(graph.create_png())

    # save the image file
    img_outfile = f'tree_dengue_{model_indx+1}{tag}.png'

    with open(save_path/img_outfile, "wb") as png:
        png.write(img.data)

    imgs += [img]
    
    model_classifiers += [clf]
    model_preds += [Y_pred]

    model_scores += [score]
    

# OVERALL METRICS

# Overall Results

# recalls defined in Parameters section

# For each recall
    # Get results for overall validation set for each model
    # Sum confusion matrices across models
    # return metrics for overall results of all models

for rec in recalls:
    confs = []
    for model_indx in range(len(dfs)):
        result_prob = model_preds[model_indx]
        label = model_valid_Ys[model_indx]
        recall = rec

        conf, roc_auc = conf_and_auc(label, result_prob, recall)

        confs += [conf]

    conf = sum(confs)
    results = pyfiles.autometrics.return_metrics(conf,roc_auc)
    result_names = ['roc_auc', 'PPV', 'NPV', 'F1', 'accuracy', 'sensitivity', 'specificity', 'odds_ratio', 'TN', 'FP', 'FN', 'TP', 'length']

# show ROC curve for each model

tprs = []
fprs = []
tprs_interp = []
base_fpr = np.linspace(0, 1, 101)

for model_indx in range(len(dfs)):
    result_prob = model_preds[model_indx]
    label = model_valid_Ys[model_indx]

    score = np.array([result_prob[j] for j in range(result_prob.shape[0])])
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot ROC Curve for Each Model
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='navy',
             lw=lw, label=f'ROC curve (area = {roc_auc: 0.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('(1–Specificity) - False Positive Rate')
    plt.ylabel('Sensitivity - True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (Model {model_indx+1})')
    plt.legend(loc="lower right")
#     plt.show()
    
    # Store fpr and tpr for combined plot
#     plt.plot(fpr, tpr, 'b', alpha=0.15)
    tprs += [tpr]
    fprs += [fpr]
    tpr[0] = 0.0
    tprs_interp += [interp(base_fpr, fpr, tpr)]
    
# Create and Save Overall ROC Plot

img_filename = f'DT_dengue_subgroup_analysis_{splits}_fold_ROC{tag}.png'

lw = 2
plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.15)
# plt.axes().set_aspect('equal', 'datalim')

for model_indx in range(len(dfs)):
    fpr = fprs[model_indx]
    tpr = tprs[model_indx]
    plt.plot(fpr, tpr, color='blue',lw=lw, alpha=0.14)
    
mean_tprs = np.array(tprs_interp).mean(axis=0)
std = np.array(tprs_interp).std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, 'black', lw=2,label=f'ROC curve (area = {roc_auc: 0.2f})')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)


plt.xlabel('(1–Specificity) - False Positive Rate')
plt.ylabel('Sensitivity - True Positive Rate')
plt.title(f'Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# save file
plt.savefig(save_path/img_filename, bbox_inches='tight')

# plt.show()
    

# SUBGROUP ANALYSIS
    
# Performs subgroup analysis for each model at given sensitivities (recalls)
# Then calculates overall metrics for each subgroup

# recalls defined in Parameters section

# For each recall, 
#     for each model's validation set
#         get confusion matrix of subgroups
#     sum confusion matrices across models for each subgroup
#     calculate metrics from summed confusion matrices
#     store metrics into a dataframe
#     write to csv file


for rec in recalls:    
    confs = 0
    for model_indx in range(len(dfs)):
        dataframes = dfs[model_indx] # subgroups for each model
        Y_pred = model_preds[model_indx]
        valid_Y = model_valid_Ys[model_indx] 
        clf = model_classifiers[model_indx]
        valid_df = subsets[model_indx]

        analysis = subgroup_analysis(clf=clf, dfs=dataframes+[valid_df], dfs_names=dfs_names+['valid_df (overall)'], dep_var='lab_result',recalls=[rec])
#         # Note: first index of analysis is recall value index
#         # second index of analysis is 0 for recall value, 1 for data corresponding to that recall

#         # Example:
#         # if recalls = [0.85,0.90,0.95]
#         # then, analysis[0][0] returns 0.85
#         # analysis[0][1] is the results for recall = 0.85, in dataframe format
#         # analysis[1][0] returns 0.90, analysis[1][1] returns results for recall=0.90, and so forth


        # Sum confusion matrices across models and store roc_auc
        rocs = analysis[0][1]['roc_auc'].values
        confs += analysis[0][1].to_numpy()[:,-4:]

        
    # Calculate Overall Metrics for Each Subgroup
    
    rocs = rocs.tolist()
    confs = confs.tolist()

    names = dfs_names+['valid_df (overall)']
    cols = ['dataframe', 'roc_auc', 'PPV', 'NPV', 'F1', 'accuracy', 'sensitivity', 'specificity', 'odds_ratio', 'TN', 'FP', 'FN', 'TP', 'size']

    results = []
    for i in range(len(names)):
        conf = [confs[i][0],confs[i][1],confs[i][2],confs[i][3]]
        res = pyfiles.autometrics.calc_metrics(confs[i][0],confs[i][1],confs[i][2],confs[i][3], rocs[i])
        results += [[names[i]] + res]

    final_subgroup_results = pd.DataFrame(results,columns=cols)

    to_file = f'DT_dengue_subgroup_analysis_folds_{splits}_recall_{rec:0.2f}{tag}.csv'
#     path = Path('../mydata/myoutput/temp')

    final_subgroup_results.to_csv(save_path/to_file, index=None)

