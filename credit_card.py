# Import the Libraries
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import streamlit as st 
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
import plotly.graph_objs as go
import os.path
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from dotenv import load_dotenv
import os
import itertools
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
load_dotenv()

### check if run locally or in the cloud, the cloud can't handle the large dataset (performance issue)
LOCAL = os.getenv('LOCAL')

if LOCAL == 'TRUE':
    pass

### helper function for cloud runtime
def export_to_pickle(data, filename):
    # Export the Model, Dataframe or Fig to Pickle
    print(f'Exporting {filename} to pickle')
    with open(f'cloud/{filename}', 'wb') as file:
        pickle.dump(data, file)

@st.cache_data #for caching the data in streamlit
def get_dataset():
     # Load the data
    # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
    df = pd.read_csv('creditcard.csv')
    export_to_pickle(df.head(), 'creditcard_head.pkl')
    return df[0:30000]

@st.cache_data #for caching the data in streamlit
def credit_card_tsne(df):
    filename = "ml/credit_card_tsne.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            credit_card = pickle.load(f)
    else:
        # Extract the features we want to use for visualization
        X = df.loc[:,df.columns!='Class']
        tsne = TSNE(n_components=3, random_state=42)
        credit_card = tsne.fit_transform(X)

        with open(filename, "wb") as f:
            pickle.dump(credit_card, f)

    return credit_card

def create_3d_visualization(X_tsne, df):
    # Create a Trace object for each class label
    classes = df['Class'].unique() # 0 and 1
    traces = []
    for c in classes:
        if c == 0:
            name = 'Normal'
        else:
            name = 'Fraud'
        mask = df['Class'] == c
        color = 'red' if c == 1 else 'blue'
        trace = go.Scatter3d(x=X_tsne[mask, 0], y=X_tsne[mask, 1], z=X_tsne[mask, 2], mode='markers',
                             marker=dict(color=df['Class'], colorscale=[[0, color], [1, color]], size=5, opacity=0.8),
                             name=str(name))
        traces.append(trace)

    # Add the traces to the data list of the Figure object
    fig = go.Figure(data=traces)

    # Set the layout parameters, including the legend
    fig.update_layout(scene=dict(xaxis_title='t-SNE Dimension 1', yaxis_title='t-SNE Dimension 2',
                                  zaxis_title='t-SNE Dimension 3'),
                      title='t-SNE Visualization of Email Features', showlegend=True)
    
    export_to_pickle(fig, "3d_visualization_dataframe_cc.pkl")
    return fig


def create_oneclass_svm_predict(df, kernel, nu, gamma, degree=3, outlier_fraction=False, gamma_scale=False):
    # separate the features and Class columns
    X = df.loc[:,df.columns!='Class']
    y = df['Class']

    if gamma_scale:
        gamma = 'scale'
    
    if outlier_fraction:
        nu = len(df[df['Class']==1])/float(len(df[df['Class']==0]))
        print(f'Outlier Fraction: {outlier_fraction}')

    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)

    clf.fit(X)

    # evaluate the model using cross-validation
    # scoring = {'accuracy': make_scorer(accuracy_score),
    #            'precision_macro': make_scorer(precision_score, average='macro',zero_division=1),
    #            'recall_macro': make_scorer(recall_score, average='macro', zero_division=1),
    #            'f1_macro': make_scorer(f1_score, average='macro', zero_division=1),
    #            'roc_auc': make_scorer(roc_auc_score, average='macro')}
    
    # cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)

    # return the average scores from the cross-validation
    y_pred = clf.predict(X) # use the model to detect outliers (cc)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # the result
    accuracy = accuracy_score(y,y_pred)
    precision = precision_score(y,y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    auc_roc = roc_auc_score(y,y_pred)

    # calculate the mean scores across all folds
    cv_accuracy = 0
    cv_precision = 0
    cv_recall = 0
    cv_f1 = 0
    cv_auc_roc = 0

    return y_pred, [cv_accuracy, cv_precision, cv_recall, cv_f1, cv_auc_roc], [accuracy, precision, recall, f1, auc_roc]

def visualize_onclasssvm(df, y_pred):
    X_tsne = credit_card_tsne(df)

    df['predicted'] = y_pred
    conditions = [(df['Class'] == 0) & (df['predicted'] == 0),
    (df['Class'] == 0) & (df['predicted'] == 1),
    (df['Class'] == 1) & (df['predicted'] == 1),
    (df['Class'] == 1) & (df['predicted'] == 0)
    ]

    choices = [0, 1, 2, 3]

    df['result'] = np.select(conditions, choices, default=np.nan)

    # Create a Trace object for each class label
    classes = df['result'].unique() # 0 and 1
    traces = []
    for c in classes:
        mask = df['result'] == c
        if c == 0:
            color = 'blue'
            name = 'Non Fraud CC Correctly Classified'
        elif c == 1:
            color = 'red'
            name = 'Non Fraud CC Incorrectly Classified'
        elif c == 2:
            color = 'green'
            name = 'Fraud CC Correctly Classified'
        else:
            color = 'orange'
            name = 'Fraud CC Incorrectly Classified'

        
        trace = go.Scatter3d(x=X_tsne[mask, 0], y=X_tsne[mask, 1], z=X_tsne[mask, 2], mode='markers',
                             marker=dict(color=color, colorscale=[[0, color], [1, color]], size=5, opacity=0.8),
                             name=str(name))
        traces.append(trace)

    # Add the traces to the data list of the Figure object
    fig = go.Figure(data=traces)

    # Set the layout parameters, including the legend
    fig.update_layout(scene=dict(xaxis_title='t-SNE Dimension 1', yaxis_title='t-SNE Dimension 2',
                                  zaxis_title='t-SNE Dimension 3'),
                      title='t-SNE Visualization of the Predicted CC Features', showlegend=True)
    
    return fig

def export_and_visualize_oneclasssvm(df, kernel, nu, gamma, degree, outlier_fraction, gamma_scale):
    y_pred, cv_res, res = create_oneclass_svm_predict(df, kernel, nu, gamma, degree, outlier_fraction, gamma_scale)
    fig = visualize_onclasssvm(df, y_pred)
    cv_values = {'Accuracy': cv_res[0], 'Precision': cv_res[1], 'Recall': cv_res[2], 'F1 Score': cv_res[3], 'AUC ROC': cv_res[4]}
    values = {'Accuracy': res[0], 'Precision': res[1], 'Recall': res[2], 'F1 Score': res[3], 'AUC ROC': res[4]}
    settings = {'kernel': kernel, 'nu': nu, 'gamma': gamma, 'degree': degree}

    export_to_pickle(fig, f"visualize_onclasssvm_{kernel}_cc.pkl")
    export_to_pickle(cv_values, f"cv_values_onclasssvm_{kernel}_cc.pkl")
    export_to_pickle(values, f"values_onclasssvm_{kernel}_cc.pkl")
    export_to_pickle(settings, f"settings_onclasssvm_{kernel}_cc.pkl")

    return fig, values, cv_values