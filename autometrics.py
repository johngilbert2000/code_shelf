# Import modules

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

from IPython.display import display_html
from bs4 import BeautifulSoup

# Use metrics_and_conf to get all metrics (in dataframe format) for a single dataset
# Use show_metrics to calculate and display metrics for a single dataset, comparing a series of recalls (in jupyter notebook)
# Use subgroup_analysis on a list of dataframes (subgroups) to obtain a dataframe containing all of their metrics (obtained with a given pretrained sklearn classifier)


# Metric Functions



def conf_and_auc(label, result_prob, min_recall):
    "Returns a confusion matrix and AUC"
    # label: validation dependent variable (test_Y)
    # result_prob: prediction of labels, 1d array (Y_pred = clf.predict(test_X) # Y_pred = clf.predict_proba(test_X)[:,1])
    # min_recall = sensitivity
    
    ### Obtaining ROC_AUC
    
    score = np.array([result_prob[j] for j in range(result_prob.shape[0])])
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    
    ### Obtaining Confusion Matrix
    
    precision, recall, thresholds = metrics.precision_recall_curve(label, score, pos_label=1)
    recall = np.asarray(recall)
    idx = (np.abs(recall - min_recall)).argmin() # Find nearest threshold
    thresh = thresholds[idx]
    
    predict_label = [1 if result_prob[j] >= thresh else 0 for j in range(result_prob.shape[0])]
    conf_mat = confusion_matrix(label, predict_label)
#     tn, fp, fn, tp = confusion_matrix(label, predict_label).ravel()
    
    # Predicted
    # by Actual
    
    # TN | FP
    # -------
    # FN | TP
    
    
    return conf_mat,roc_auc

    
def find_metric_results(conf_mat, roc_auc):
    "Find metrics using the Confusion Matrix"
    
    # Confusion Matrix was originally in this form:
    # TN | FP
    # -------
    # FN | TP
    
    TN = conf_mat[0,0] # true negative
    FP = conf_mat[0,1] # false positive
    FN = conf_mat[1,0] # false negative
    TP = conf_mat[1,1] # true positive
    
    PPV = TP / (TP + FP) if (TP+FP != 0) else 0 # positive predict value
    NPV = TN / (TN + FN) if (TN+FN != 0) else 0 # negative predict value
    F1 = 2*TP / (2*TP + FP + FN) #
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP /(TP + FN) if (TP+FN != 0) else 0
    specificity = TN /(TN + FP) if (TN+FP != 0) else 0
    odds_ratio = (TP * TN) /(FP * FN) if (FP*FN != 0) else 0
    
    return [roc_auc, PPV, NPV, F1, accuracy, sensitivity, specificity, odds_ratio]


def return_metrics(conf_mat, roc_auc):
    "Find metrics using the Confusion Matrix"
    
    # Confusion Matrix was originally in this form:
    # TN | FP
    # -------
    # FN | TP
    
    TN = conf_mat[0,0] # true negative
    FP = conf_mat[0,1] # false positive
    FN = conf_mat[1,0] # false negative
    TP = conf_mat[1,1] # true positive
    total = sum([TN,FP,FN,TP])
    
    epsilon = 0.0
    
    PPV = TP / (TP + FP) if (TP+FP != 0) else 0 # positive predict value
    NPV = TN / (TN + FN) if (TN+FN != 0) else 0 # negative predict value
    F1 = 2*TP / (2*TP + FP + FN) #
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP /(TP + FN) if (TP+FN != 0) else 0
    specificity = TN /(TN + FP) if (TN+FP != 0) else 0
    odds_ratio = (TP * TN) /(FP * FN) if (FP*FN != 0) else 0
    
    return [roc_auc, PPV, NPV, F1, accuracy, sensitivity, specificity, odds_ratio, TN, FP, FN, TP, total]

def calc_metrics(TN,FP,FN,TP, roc_auc):
    "Find metrics using the Confusion Matrix"
    
    # Confusion Matrix was originally in this form:
    # TN | FP
    # -------
    # FN | TP
    
#     TN = conf_mat[0,0] # true negative
#     FP = conf_mat[0,1] # false positive
#     FN = conf_mat[1,0] # false negative
#     TP = conf_mat[1,1] # true positive
    total = sum([TN,FP,FN,TP])
    
    PPV = TP / (TP + FP) if (TP+FP != 0) else 0 # positive predict value
    NPV = TN / (TN + FN) if (TN+FN != 0) else 0 # negative predict value
    F1 = 2*TP / (2*TP + FP + FN) #
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP /(TP + FN) if (TP+FN != 0) else 0
    specificity = TN /(TN + FP) if (TN+FP != 0) else 0
    odds_ratio = (TP * TN) /(FP * FN) if (FP*FN != 0) else 0
    
    return [roc_auc, PPV, NPV, F1, accuracy, sensitivity, specificity, odds_ratio, TN, FP, FN, TP, total]


def metric_df(metric_results, conf_mat):
    "Converts metrics and confusion matrix into dataframes"

    # metric_results: 1d array of metric values
    # conf_mat: 2x2 np.array of confusion matrix

    metric_labels = ['roc_auc','PPV','NPV','F1','accuracy','sensitivity','specificity','odds_ratio']
    percent_indxs = [1,2,4,5,6] # value indices for values to be displayed as a percentage

    # Convert some vals to percent
    for i in percent_indxs:
        metric_results[i] = metric_results[i]*100
        metric_results[i] = '%s' % str('%.4f' % metric_results[i]) + '%'

    metrics = zip(metric_labels, metric_results)
    
    metric_df = pd.DataFrame(data=metrics, columns=['Metric','Result'])
    conf_mat_df = pd.DataFrame(data=conf_mat, columns=['Predicted False','Predicted True'])
    
    # Confusion Matrix
    # Predicted
    # by Actual
    
    # TN | FP
    # -------
    # FN | TP
    
#     conf_mat_df.insert(loc=0, column='_', value=['Actual False','Actual True'], allow_duplicates=False)
    
    return metric_df, conf_mat_df



# Find All Metrics

def metrics_and_conf(label, result_prob, min_recall):
    "Obtain all metrics and the confusion matrix as two dataframes"
    # label: validation dependent variable (test_Y)
    # result_prob: prediction of labels, 1d array (Y_pred = clf.predict(test_X) # Y_pred = clf.predict_proba(test_X)[:,1])
    # min_recall = sensitivity
    
    # Obtain Confusion Matrix and AUC
    conf_mat, roc_auc = conf_and_auc(label, result_prob, min_recall)
    
    # Calculate Metrics with Confusion Matrix: ['roc_auc','PPV','NPV','F1','accuracy','sensitivity','specificity','odds_ratio']
    metric_results = find_metric_results(conf_mat, roc_auc)
    
    # Convert Metrics and Confusion Matrix into Dataframes
    metric_results, conf_mat = metric_df(metric_results, conf_mat)
    
    return metric_results, conf_mat




# Display Results


def display_together(dfs, captions=[], spacing=20, *args):
    "Displays pandas dataframes together in Jupyter Notebook"
#     dfs: list of dataframes to display
#     captions: list of captions for each dataframe
#     spacing: spaces between dataframes when displayed
    
    
    # If captions are none, must create an array of empty captions
    if len(captions) < len(dfs):
        missing = len(dfs) - len(captions)
        captions += ['']*missing
    
    html_str = ''
    
    # Obtain HTML for each dataframe
    for i in range(len(dfs)):
        
        df_styler = dfs[i].reset_index(drop=True).style.\
                set_table_attributes("style='display:inline'").\
                set_caption(captions[i])
        
        df_styler = df_styler.hide_index()
        html_str += df_styler._repr_html_()
        html_str += '&nbsp;' * spacing
    html_str += '<br>'*2

    # Hide df labels with BeautifulSoup
    soup = BeautifulSoup(html_str, 'html.parser')

    tags = soup.find_all('th')
    for tag in tags:
        tag['hidden']=True
        
    html_str = str(soup)
    
    # Display HTML
    display_html(html_str, raw=True)
    
def display_together_with_labels(dfs, captions=[], spacing=36, *args):
    "incomplete: don't use this function"
    
    
    "Displays pandas dataframes together in Jupyter Notebook"
#     dfs: list of dataframes to display
#     captions: list of captions for each dataframe
#     spacing: spaces between dataframes when displayed
    
    # If captions are none, must create an array of empty captions
    if len(captions) < len(dfs):
        missing = len(dfs) - len(captions)
        captions += ['']*missing
    
    html_str = ''
    
    # Obtain HTML for each dataframe
    for i in range(len(dfs)):
        
        df_styler = dfs[i].reset_index(drop=True).style.\
                set_table_attributes("style='display:inline'").\
                set_caption(captions[i])
        
#         df_styler = df_styler.hide_index()
        html_str += df_styler._repr_html_()
        html_str += '&nbsp;' * spacing
    html_str += '<br>'*2
    
    # Hide df labels with BeautifulSoup
    soup = BeautifulSoup(html_str, 'html.parser')

    tags = soup.find_all('col1')
    for tag in tags:
        tag['hidden']=False
        tag['style']='bold'
        
    html_str = str(soup)
    
    # Display HTML
    display_html(html_str, raw=True)

    
def show_preprocessed_metrics(mets, conf_mats, recalls=[]):
    "Uses display_together to show metrics in table form"
    
    # mets: list of metric_results in dataframe format
    # conf_mats: list of confusion matrices in dataframe format
    # sensitivities: list of sensitivity captions in float format
    
    # Create Captions
    sens_cap = f''
    for rec in recalls:
        sens_cap += f'sensitivity={rec:.2f},'
    sens_cap = sens_cap.split(',')[:-1]
    conf_cap = ['Confusion_Matrix']*len(recalls)

    display_html('<br><b>Metrics</b>',raw=True)
    display_together(dfs=mets, captions=sens_cap)
    display_together(spacing=37, dfs=conf_mats, captions=conf_cap)

    
def show_metrics(label, result_prob, recalls=[], title='Metrics',conf_display=True, hide_display=False):
    "Uses display_together and metrics_and_conf to calculate and show metrics in table form"
    
    # mets: list of metric_results in dataframe format
    # conf_mats: list of confusion matrices in dataframe format
    # recalls: list of sensitivities in float format
    
    # title: string
    # conf_display: boolean; if true, describe how the confusion matrix is shown
    # hide_display: boolean; turn off HTML display (and just return metrics and confusion matrix as dataframes)
    
    
    # Calculate Metrics and Confusion Matrix for each Recall
    conf_mats = []
    mets = []
    for i in range(len(recalls)):
        metric_results, conf_mat = metrics_and_conf(label=label, result_prob=result_prob, min_recall=recalls[i])
        mets += [metric_results]
        conf_mats += [conf_mat]
    
    # Create Captions
    sens_cap = f''
    for rec in recalls:
        sens_cap += f'sensitivity={rec:.2f},'
    sens_cap = sens_cap.split(',')[:-1]
    conf_cap = ['Confusion_Matrix']*len(recalls)
    
    # Modify Spacing
    n = 0
    if len(recalls) > 4:
        n = 10
    if len(recalls) > 5:
        n = 18
    
    if conf_display:
        print("Note: Confusion Matrices are of the form:\n \
        \n                 Predicted \
        \n Actual  [ true_neg   false_pos ] \
        \n         [ false_neg  true_pos  ]")

    # Display Results
    
    if not hide_display:
        display_html(f'<br><b>{title}</b>',raw=True)
        display_together(spacing=20-n, dfs=mets, captions=sens_cap)
        display_together(spacing=36-n, dfs=conf_mats, captions=conf_cap)
    
    return metric_results, conf_mat
    
    

def show_metrics_new(label, result_prob, recalls=[], title='Metrics',conf_display=True, hide_display=False):
    "Uses display_together and metrics_and_conf to calculate and show metrics in table form"
    
    # mets: list of metric_results in dataframe format
    # conf_mats: list of confusion matrices in dataframe format
    # recalls: list of sensitivities in float format
    
    # title: string
    # conf_display: boolean; if true, describe how the confusion matrix is shown
    # hide_display: boolean; turn off HTML display (and just return metrics and confusion matrix as dataframes)
    
    
    # Calculate Metrics and Confusion Matrix for each Recall
    conf_mats = []
    mets = []
    for i in range(len(recalls)):
        metric_results, conf_mat = metrics_and_conf(label=label, result_prob=result_prob, min_recall=recalls[i])
        mets += [metric_results]
        conf_mats += [conf_mat]
    
    # Create Captions
    sens_cap = f''
    for rec in recalls:
        sens_cap += f'sensitivity={rec:.2f},'
    sens_cap = sens_cap.split(',')[:-1]
    conf_cap = ['Confusion_Matrix']*len(recalls)
    
    # Modify Spacing
    n = 0
    if len(recalls) > 4:
        n = 10
    if len(recalls) > 5:
        n = 18
    
    if conf_display:
        print("Note: Confusion Matrices are of the form:\n \
        \n                 Predicted \
        \n Actual  [ true_neg   false_pos ] \
        \n         [ false_neg  true_pos  ]")

    # Display Results
    
    if not hide_display:
        display_html(f'<br><b>{title}</b>',raw=True)
        display_together(spacing=20-n, dfs=mets, captions=sens_cap)
        display_together(spacing=36-n, dfs=conf_mats, captions=conf_cap)
    
    return mets, conf_mats




def subgroup_analysis(clf, dfs, dfs_names, dep_var, recalls=[0.85,0.90,0.95]):
    "Get all metrics for a list of subgroups, assessed with a pretrained sklearn classifier"
    # clf: classifier (pretrained model)
    # dfs: list of dataframes to be assessed by classifier 
    # dfs_names: list of names for each df in dfs
    # dep_var: name dependent variable, used to split inputs from outputs
    # recalls: list of sensitivities to be assessed for each df in dfs
    
    # Note on return value:
    # first index of analysis corresponds to each recall in recalls
    # second index of analysis is 0 for recall value, 1 for data corresponding to that recall
    
    # Prepare data & get predictions
    results = []
    for i in range(len(dfs)):
        dataframe = dfs[i]
        name = dfs_names[i]
        length_df = len(dfs[i])
        

        # Separate inputs (X) from outputs (Y)
        valid_X = dataframe.loc[:,dataframe.columns != dep_var]
        valid_Y = dataframe.loc[:,dataframe.columns == dep_var]

        # Convert to numpy arrays
        valid_X = valid_X.to_numpy('float64')
        valid_Y = valid_Y.to_numpy('float64').flatten()

#         Y_pred = clf.predict_proba(valid_X)[:,1]
        Y_pred = clf.predict(valid_X)
        score = clf.score(valid_X, valid_Y)

        results += [(name, Y_pred, valid_Y, score, length_df)]
        
        # for rec in recalls ... [refactor code here, if desired]
        
    # Obtain metrics
    mets_with_recs = [] # all metrics with corresponding recall value
    
    for rec in recalls:
        
        all_mets = []
        for i in range(len(dfs)):
            res = results[i]
            name = res[0]
            Y_pred = res[1]
            valid_Y = res[2]
            length_df = res[4]
            
            metric_results, conf_mat = metrics_and_conf(label=valid_Y, result_prob=Y_pred, min_recall=rec)
    
            # transpose metrics and organize data (technical debt...)
            col_names = metric_results.T.to_numpy()[0].tolist()
            metric_results = metric_results.T.to_numpy()[1].tolist()
            conf_mat = conf_mat.to_numpy().flatten().tolist()
    
            col_names = ['Dataframe','size'] + col_names + ['true_neg','false_pos','false_neg','true_pos']
            
            all_mets += [[name,length_df] + metric_results + conf_mat]
            
        df_all_metrics = pd.DataFrame(data=all_mets, columns=col_names)
        mets_with_recs += [(rec,df_all_metrics)]
        
    return mets_with_recs
            
