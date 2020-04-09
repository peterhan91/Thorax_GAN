import numpy as np
import os
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
from math import pi
import seaborn as sns
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def determine_cutoffs(test_df, pred_df, cutoff_num=10000):
    cutoffs = np.linspace(0.0, 1.0, num=cutoff_num)
    cutoff_result = pd.DataFrame(columns=test_df.columns[1:], index=range(1))
    
    for pathology in test_df.columns[1:]:
        raw_effusions = np.array(pred_df['prob_'+pathology].values.tolist())
        true_effusion = np.array(list(map(int, test_df[pathology].values.tolist())))
        SS_result = []

        for cutoff in cutoffs:
            pred_effusions = (raw_effusions > cutoff).astype(np.int)
            confusion = metrics.confusion_matrix(true_effusion, pred_effusions)
            TN, FP = confusion[0, 0], confusion[0, 1]
            FN, TP = confusion[1, 0], confusion[1, 1]
            spec = TN / (TN + FP)
            sens = TP / (TP + FN)
            result = (1-spec)*(1-spec) + (1-sens)*(1-sens)
            SS_result.append(result)

        if len(SS_result) == len(cutoffs):
            index, _ = min(enumerate(SS_result), key=operator.itemgetter(1))
            cutoff_result[pathology][0] = cutoffs[index]
    
    return cutoff_result


def bootstrapping (true_df, pred_df, cutoff_df, save_dir, save_raw=True, repeat=10000, sub_num=10000):
    
    pred_imglist = pred_df['Image Index'].tolist()
    true_imglist = true_df['Image Index'].tolist()
    if pred_imglist == true_imglist:
        concate_df = pd.concat([pred_df, true_df], axis=1)
    else:
        raise ValueError('true_df image index is different from pred_df index!')
        
    AUC_result = pd.DataFrame(columns=true_df.columns[1:], index=range(repeat))
    accuracy_result = pd.DataFrame(columns=true_df.columns[1:], index=range(repeat))
    PPV_result = pd.DataFrame(columns=true_df.columns[1:], index=range(repeat))
    NPV_result = pd.DataFrame(columns=true_df.columns[1:], index=range(repeat))
    sensitivity_result = pd.DataFrame(columns=true_df.columns[1:], index=range(repeat))
    specificity_result = pd.DataFrame(columns=true_df.columns[1:], index=range(repeat))
    F1_result = pd.DataFrame(columns=true_df.columns[1:], index=range(repeat))
    
    for i in range(repeat):
        sub_df = concate_df.sample(n=sub_num, replace=True, random_state=i)
        for pathology in true_df.columns[1:]:
            y_true = sub_df[pathology].values.tolist()
            y_true = list(map(int, y_true))
            y_pred = sub_df['prob_'+pathology].values.tolist()
            AUC_result[pathology][i]=roc_auc_score(y_true, y_pred)
            
            threshold = cutoff_df[pathology].values.tolist()[0]
            y_pred_thres = np.array((y_pred > threshold).astype(np.int))
            y_true_thres = np.array(y_true)
            accuracy_result[pathology][i] = accuracy_score(y_true_thres, y_pred_thres)
            
            confusion = metrics.confusion_matrix(y_true_thres, y_pred_thres)
            TN, FP = confusion[0, 0], confusion[0, 1]
            FN, TP = confusion[1, 0], confusion[1, 1]
            sens = TP / (TP + FN)
            spec = TN / (TN + FP)
            ppv = TP / (TP + FP)
            npv = TN / (TN + FN)
            f1 = 2*TP / (2*TP + FP + FN)
            sensitivity_result[pathology][i] = sens
            specificity_result[pathology][i] = spec
            PPV_result[pathology][i] = ppv
            NPV_result[pathology][i] = npv
            F1_result[pathology][i] = f1
    if save_raw:
        AUC_result.to_csv(os.path.join(save_dir, 'auc.csv'), index=False)
        accuracy_result.to_csv(os.path.join(save_dir, 'accuracy.csv'), index=False)
        sensitivity_result.to_csv(os.path.join(save_dir, 'sensitivity.csv'), index=False)
        specificity_result.to_csv(os.path.join(save_dir, 'specificity.csv'), index=False)
        PPV_result.to_csv(os.path.join(save_dir, 'ppv.csv'), index=False)
        NPV_result.to_csv(os.path.join(save_dir, 'npv.csv'), index=False)
        F1_result.to_csv(os.path.join(save_dir, 'f1.csv'), index=False)

def mean_std(pathology_list):
    column_list = ['auc', 'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    mean_df = pd.DataFrame(columns=column_list, index=pathology_list)
    std_df = pd.DataFrame(columns=column_list, index=pathology_list)
    for csv in column_list:
        result_df = pd.read_csv(csv+'.csv')
        for pathology in pathology_list:
            result = np.array(result_df[pathology].values.tolist())
            mean = np.mean(result)
            std = np.std(result)
            mean_df.loc[pathology, csv] = mean
            std_df.loc[pathology, csv] = std
            
    mean_df.to_csv('result_mean.csv', index=True)
    std_df.to_csv('result_std.csv', index=True)

def plot_hist(pathology_list, num_bins=40):
    csv_list = ['auc', 'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    for csv in csv_list:
        result_df = pd.read_csv(csv+'.csv')
        fig, ax = plt.subplots(figsize=(12,10))
        ax.set_xlim([0, 1.0])
        ax.set_xlabel(csv, fontsize=16, fontname="Calibri")
        ax.set_ylabel('Counts', fontsize=16, fontname="Calibri")
        for pathology in pathology_list:
            _ = ax.hist(result_df[pathology], num_bins, alpha=0.3, label=pathology)
        ax.legend()
        plt.savefig(csv+'.png', dpi=800, bbox_inches = 'tight', pad_inches = 0)


xray_df = pd.read_csv('/media/tianyu.han/mri-scratch/DeepLearning/CheXnet/Pytorch_original/nih_labels.csv')
test_df, rest_df = [x for _, x in xray_df.groupby(xray_df['fold'] != 'test')]
test_df = test_df.drop(columns=['Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender',
                               'View Position', 'fold', 'Emphysema', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
                               'Pleural_Thickening', 'Fibrosis'])
true_df = test_df.reset_index(drop=True)
pred_df = pd.read_csv('preds.csv')
pred_df['Image Index']=pred_df['Image Index'].apply(lambda x: os.path.basename(x))
cutoff_df = determine_cutoffs(true_df, pred_df)
bootstrapping(true_df, pred_df, cutoff_df, save_dir=os.getcwd())
mean_std(true_df.columns[1:])
plot_hist(true_df.columns[1:])