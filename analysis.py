# Code for the paper: Kaissis G. et al.: A machine learning algorithm predicts molecular subtypes in pancreatic ductal adenocarcinoma with differential response to gemcitabine-based versus FOLFIRINOX chemotherapy
# Copyright Kaissis G., 2019
# Version 2019.7
# This work is available under the GNU Affero General Public License v3.0
# If code is executed inside a REPL or non-interactive Python environment, please add plt.show() rendering calls
# Please refer to README.MD for instructions on how to use the code

from __future__ import print_function, absolute_import, division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, permutation_test_score, train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, make_scorer, confusion_matrix, roc_auc_score
import statistics
from xgboost import XGBClassifier as xgb
import joblib


#Loading features and labels as dataframes using the joblib format
#features=joblib.load("features.dat")
#labels=joblib.load("labels.dat")

#Loading features and labels as csv files from pandas
#features=pd.read_csv("Features.csv")
#labels=pd.read_csv("Labels.csv")

#custom scoring functions
def sensitivity(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sens=tp/(tp+fn)
    return sens


def specificity(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spec=tn/(tn+fp)
    return spec

sss=StratifiedShuffleSplit(n_splits=10, test_size=.3, random_state=122)
tree=xgb(reg_alpha=1, learning_rate=0.01, random_state=122)

auc=cross_val_score(tree, X=features, y=labels.values.ravel(), cv=sss, scoring="roc_auc", n_jobs=-1)

sens=cross_val_score(tree, X=features, y=labels.values.ravel(), cv=sss, scoring=make_scorer(sensitivity), n_jobs=-1)

spec=cross_val_score(tree, X=features, y=labels.values.ravel(), cv=sss, scoring=make_scorer(specificity), n_jobs=-1)

print(f"AUC: {auc.mean():.2f}±{auc.std():.2f}")
print(f"Sensitivity: {sens.mean():.2f}±{sens.std():.2f}")
print(f"Specificity: {spec.mean():.2f}±{spec.std():.2f}")

#Permutation scoring
perm_auc_score, _, auc_p=permutation_test_score(tree, X=features, y=labels.values.ravel(), cv=sss, n_permutations=100, n_jobs=-1, verbose=0, scoring="roc_auc")
perm_sens, _, sens_p=permutation_test_score(tree, X=features, y=labels.values.ravel(), cv=sss, n_permutations=100, n_jobs=-1, verbose=0, scoring=make_scorer(sensitivity))
perm_spec, _, spec_p=permutation_test_score(tree, X=features, y=labels.values.ravel(), cv=sss, n_permutations=100, n_jobs=-1, verbose=0, scoring=make_scorer(specificity))

print(f"Permutation AUC: {perm_auc_score:.2f}, P={auc_p:.3f}")
print(f"Permutation Sensitivity: {perm_sens:.2f}, P={sens_p:.3f}")
print(f"Permutation Specificity: {perm_spec:.2f}, P={spec_p:.3f}")

#Feature Importance
tree.fit(features, labels.values.ravel())
feature_df=pd.DataFrame({"Feature":features.columns, "Importance":tree.feature_importances_})
sorted_features=feature_df[feature_df["Importance"]>0].sort_values(by="Importance", ascending=False)
sorted_features # DataFrame of the features with importance >0 sorted by importance

#ROC-AUC plot
fpr_list=[]
tpr_list=[]
roc_auc_list=[]
for i in range(10):
    X_train, X_test, y_train, y_test=train_test_split(features, labels.values.ravel(), test_size=0.30, stratify=labels, shuffle=True)
    tree.fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, tree.predict_proba(X_test)[:,1], pos_label=1)
    roc_auc=roc_auc_score(y_test, tree.predict_proba(X_test)[:,1])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)


av_tpr=[pd.DataFrame(tpr_list)[i].mean() for i in pd.DataFrame(tpr_list).columns]
avg_fpr=[pd.DataFrame(fpr_list)[i].mean() for i in pd.DataFrame(fpr_list).columns]
avg_roc=statistics.mean(roc_auc_list)
roc_stdev=statistics.stdev(roc_auc_list)

with plt.style.context('seaborn-colorblind'):
    plt.figure(figsize=(8,6))
    for i in range(10):
        plt.plot(fpr_list[i], tpr_list[i], alpha=0.2, lw=1, aa=True)
    plt.plot(avg_fpr, av_tpr, lw=3, ls="dotted", color="black", label="Average ROC\nAUC={:.2f}±{:.2f}".format(auc.mean(), auc.std()))
    plt.plot((0,1), (0,1), ls="dashed", c="gray")
    plt.legend(loc="lower right")
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")

# Pearson Correlation Matrix of Selected features
selected = [
"original_firstorder_Entropy",
"gradient_firstorder_Kurtosis",
"log-sigma-1-0-mm-3D_glcm_Imc2",
"log-sigma-3-0-mm-3D_firstorder_Kurtosis",
"original_glszm_SizeZoneNonUniformityNormalized",
"wavelet-HHL_glcm_Imc2",
"wavelet-HHL_glszm_SmallAreaEmphasis",
"wavelet-HHL_glszm_ZonePercentage",
"original_shape_Maximum2DDiameterRow",
"log-sigma-2-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis",
"original_glszm_LargeAreaLowGrayLevelEmphasis",
"wavelet-HLL_glszm_ZonePercentage",
"wavelet-LHL_firstorder_Kurtosis",
 ]

c = features[selected]
c_ = c.corr(method="spearman")

plt.figure(figsize=(5,5))
sns.heatmap(c_, vmax=1, vmin=-1, center=0, cmap="viridis")
