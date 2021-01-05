
import warnings
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.cluster import hierarchy as hc
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import seaborn as sns





#Preprocess dat. Fill nans, Reduce outliers, Change categorial strings to numbers
def pre_process(K1D_features):
    c_k1d = {}
    c_k1d_df = K1D_features.copy()
    #no missing data to Age
    cols = K1D_features.columns.drop('Age','Diagnosis')
#Creating a one hot vector:
    c_k1d_df.Gender[c_k1d_df.Gender == 'Male'] = np.int(0)
    c_k1d_df.Gender[c_k1d_df.Gender == 'Female'] = np.int(1)
    c_k1d_df.Diagnosis[c_k1d_df.Diagnosis == 'Negative'] = np.int(0)
    c_k1d_df.Diagnosis[c_k1d_df.Diagnosis == 'Positive'] = np.int(1)
    c_k1d_df.replace('Yes', np.int(1), inplace=True)
    c_k1d_df.replace('No', np.int(0), inplace=True)

    # Change to nan non values:
    c_k1d_df = c_k1d_df.apply(pd.to_numeric, errors='coerce')


    # Make a dictionary and remove all nans by the probability of the array
    c_k1d = {k: [np.random.choice(v[v.notnull()].array) if np.isnan(elem) else elem for elem in v] for (k, v) in
             c_k1d_df.items()}
    c_k1d_df = pd.DataFrame(c_k1d)

    c_k1d_df.boxplot(column = 'Age')
    plt.ylabel('Age [years]')
    plt.show()
    #Age outliers are a big size of the data and we dicided not to remove them
    d_summary = c_k1d_df.Age.describe()
    index = c_k1d_df[(c_k1d_df['Age'] > d_summary.at['75%']) | (c_k1d_df['Age'] < d_summary.at['25%'])].index
    c_k1d_df.drop(index, inplace=True)
    return pd.DataFrame(c_k1d_df)


#Split data to train and test with a 20:80 split
def split_data(K1D_data_work,output):
    cols = K1D_data_work.drop(output, axis=1).columns
    Diagnosis = K1D_data_work[output]
    data = K1D_data_work.drop(output, axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(data, np.ravel(Diagnosis),
                                                        test_size=0.2, random_state=0, stratify=np.ravel(Diagnosis))
    X_train_df = pd.DataFrame(X_train, columns=cols)
    X_test_df = pd.DataFrame(X_test, columns=cols)
    return X_train, X_test, y_train, y_test,  X_train_df, X_test_df

#Visualize the data by the split distribution. seperate parameteric and attribute data
def visualizationAndExploration(K1D_data_work, X_train, X_test, y_train, y_test ):
    # Distribution of the boolean features between test and train:
    table_cols = ['positive feature', 'positive train [%]', 'positive test [%]', 'diff train test [%]']
    bool_distribution_df = pd.DataFrame(columns=table_cols)
    X_train_val = X_train.drop(['Age'],  axis=1).to_numpy()
    X_test_val = X_test.drop(['Age'],  axis=1).to_numpy()
    for idx, col in enumerate(X_train_val.transpose()):
        bool_distribution_df.loc[idx, 'positive train [%]'] = np.sum(col)*100/col.size
        bool_distribution_df.loc[idx, 'positive feature'] = K1D_data_work.columns.drop('Age')[idx]
    for idx, col in enumerate(X_test_val.transpose()):
        bool_distribution_df.loc[idx, 'positive test [%]'] = np.sum(col)*100/col.size
    bool_distribution_df['diff train test [%]'] = bool_distribution_df['positive train [%]'] - bool_distribution_df['positive test [%]']
    #Visualize the age parameter test train distribution which is not boolean:
    age_df_train = pd.DataFrame(X_train['Age'])
    age_df_test = pd.DataFrame(X_test['Age'])
    plt.hist(age_df_train.values, 20, alpha = 0.5)
    plt.hist(age_df_test.values, 20, alpha=0.5)
    plt.xlabel('Histogram Age')
    plt.ylabel('Count')
    plt.title('Blue: train mu = {0:.2f}'.format((np.mean(age_df_train.to_numpy()))) + ' train std: {0:.2f}'.format((np.std(age_df_train.to_numpy()))) + '\n' +'Orange: test mu = {0:.2f}'.format((np.mean(age_df_test.to_numpy())))+ ' test std: {0:.2f}'.format((np.std(age_df_test.to_numpy()))))
    plt.show()
    return bool_distribution_df

#Show the output label and feature relationship
def show_feature_label_relationship(K1D_data_work, Diagnosis):
    #Plot disgnosis results for all boolean features grouped to yes and no
    cols = K1D_data_work.drop(['Age', 'Diagnosis'],  axis=1).columns
    diagnosis_positive_yes = []
    diagnosis_positive_no = []
    diagnosis_negative_yes = []
    diagnosis_negative_no = []
    fig, ax = plt.subplots()
    x = np.arange(len(cols))  # the label location
    width = 0.20  # the width of the bars
    for col in cols:
        mask_p_yes = (Diagnosis['Diagnosis'] == 1).values & (K1D_data_work[col] == 1).values
        mask_p_no = (Diagnosis['Diagnosis'] == 1).values & (K1D_data_work[col] == 0).values
        diagnosis_positive_yes.append(K1D_data_work[mask_p_yes][col].size)
        diagnosis_positive_no.append(K1D_data_work[mask_p_no][col].size)
        mask_n_yes = (Diagnosis['Diagnosis'] == 0).values & (K1D_data_work[col] == 1).values
        mask_n_no = (Diagnosis['Diagnosis'] == 0).values & (K1D_data_work[col] == 0).values
        diagnosis_negative_yes.append(K1D_data_work[mask_n_yes][col].size)
        diagnosis_negative_no.append(K1D_data_work[mask_n_no][col].size)
    bar1 = ax.bar(x - 2*width, diagnosis_positive_yes, width, label='marked yes for label and diagnosed positive', color='red')
    bar2 = ax.bar(x - width, diagnosis_positive_no, width, label='marked no for label and diagnosed positive', color='red', alpha=0.3)
    bar3 = ax.bar(x, diagnosis_negative_yes, width, label='marked yes for label and diagnosed negative', color='green', alpha=0.5)
    bar4 = ax.bar(x + width, diagnosis_negative_no, width, label='marked no for label and diagnosed negative', color='green', alpha=0.3)
    ax.set_ylabel('Num of patients')
    ax.set_title('Diagnosis by label')
    ax.set_xticks(x)
    ax.set_xticklabels(cols.to_list(),rotation=90, multialignment='left')
    ax.legend(loc='upper right',bbox_to_anchor=(1.15, 1., 1., .0), mode="expand")
    plt.show()
#Plot age feature which is not boolean in 2 histograms seperated to diagnosis results
    fig, ax = plt.subplots()
    plt.hist(K1D_data_work[(Diagnosis['Diagnosis'] == 1).values]['Age'].values, 20, color='red',alpha=0.5,label='Age diagnosed positive')
    plt.hist(K1D_data_work[(Diagnosis['Diagnosis'] == 0).values]['Age'].values, 20, color='green', alpha=0.5,label='Age diagnosed negative')
    plt.xlabel('Histogram Age')
    plt.ylabel('Count')
    ax.set_title('Age feature diagnosis by label')
    ax.legend()
    plt.show()

#Show feature to feature relationship
def show_feature_2_feature_relationship(K1D_data_work):
    corr = np.round(stats.spearmanr(K1D_data_work[K1D_data_work.drop(['Diagnosis'],  axis=1).columns]).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(10, 6))
    with plt.rc_context({'lines.linewidth': 3}):
        dendrogram = hc.dendrogram(z, labels=K1D_data_work.drop(['Diagnosis'],  axis=1).columns, orientation='left', leaf_font_size=10)
    plt.xticks(fontsize='10')
    plt.yticks(fontsize='10')
    plt.xlabel('1 - Spearman correlation grade')
    plt.ylabel('Feature')
    plt.title('Spearman correlation grade')
    plt.show()


#Calculate cross validation scores
#Normelize with a min max scaler after the splits
def calc_CV(X, y, my_classifier, K):
    warnings.filterwarnings('ignore', 'Solver terminated early.*')
    kf = SKFold(n_splits=K,random_state=0,shuffle=True)
    auc_vec = np.zeros(K)
    k = 0
    scaler = MinMaxScaler()
    for train_idx, test_idx in kf.split(X, y):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_pred = my_classifier.fit(scaler.fit_transform(x_train), y_train)
        y_test_pred = my_classifier.predict(scaler.transform(x_test))
        auc_vec[k] = roc_auc_score(y_test, y_test_pred)
        k = k+1
    mu_auc = np.mean(auc_vec)
    sigma_auc = np.std(auc_vec)
    return mu_auc, sigma_auc

#Report AUC, F1, Loss, Acc scores also show confusion matrices
def report_evaluation_matrics(classifierName, classifier, X_train_scaled, X_test_scaled, y_train, y_test):
    classifier.fit(X_train_scaled, y_train)
    y_test_pred = classifier.predict(X_test_scaled)
    print(classifierName + ": AUC is: " + str("{0:.2f}".format(100 * metrics.roc_auc_score(y_test, y_test_pred))) + "%")
    print(classifierName + ": F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_test_pred, average='macro'))) + "%")
    print(classifierName + ": Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_test_pred))) + "%")
    cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    ax = plt.subplot()
    ax.set_title(classifierName+' confusion matrix')
    sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set(ylabel='True labels', xlabel='Predicted labels')
    plt.show()

#Create a random forest importance table
def random_forest_importance(classifier, data):
    return pd.DataFrame({'cols': classifier.columns, 'imp': data.feature_importances_}).sort_values('imp', ascending=False)

# Create a scatter plot
def scatter_plot(X_train_new, y_train, xlabel, ylabel, title):
    plt.scatter(X_train_new[y_train == 1, 0], X_train_new[y_train == 1, 1], marker='o', color='red', s=30,
                label='Negative')
    plt.scatter(X_train_new[y_train == 0, 0], X_train_new[y_train == 0, 1], marker='o', color='green', s=30,
                label='Positive')
    plt.xticks(fontsize='10')
    plt.yticks(fontsize='10')
    plt.xlabel(xlabel, size=10)
    plt.ylabel(ylabel, size=10)
    plt.legend(scatterpoints=1, loc='upper right', ncol=1, fontsize=10)
    plt.title(title, size=20)
    plt.show()
