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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
load_dotenv()

### check if run locally or in the cloud, the cloud can't handle the large dataset (performance issue)
LOCAL = os.getenv('LOCAL')

if LOCAL == 'TRUE':
    # Download the NLTK Data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


### helper function for cloud runtime
def export_to_pickle(data, filename):
    # Export the Model, Dataframe or Fig to Pickle
    print(f'Exporting {filename} to pickle')
    with open(f'cloud/{filename}', 'wb') as file:
        pickle.dump(data, file)

### scale the data
def scale_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)

@st.cache_data #for caching the data in streamlit
def get_dataset():
     # Load the data
    # https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset?resource=download
    df = pd.read_csv('spam_assassin.csv')
    export_to_pickle(df.head(), 'spam_dataframe_head.pkl')
    return df

@st.cache_data #for caching the data in streamlit
def spam_mail_features(df):

    # Define the Function to Calculate the Features
    def count_nouns(tokens):
        nouns = [token for token, pos in tokens if pos.startswith('N')]
        return len(nouns)

    def count_stopwords(tokens):
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        return len(tokens) - len(filtered_tokens)

    def avg_token_length(tokens):
        total_length = sum([len(token) for token in tokens])
        return total_length / len(tokens)

    def count_special_chars(text):
        pattern = r'[^\w\s]'
        special_chars = re.findall(pattern, text)
        return len(special_chars)

    def count_uppercase_words(tokens):
        uppercase_words = [token for token in tokens if token.isupper()]
        return len(uppercase_words)

    def count_adverbs(tokens):
        adverbs = [token for token, pos in tokens if pos.startswith('RB')]
        return len(adverbs)

    def count_personal_pronouns(tokens):
        personal_pronouns = [token for token, pos in tokens if pos == 'PRP' or pos == 'PRP$']
        return len(personal_pronouns)

    def count_capital_letters(text):
        capital_letters = [char for char in text if char.isupper()]
        return len(capital_letters)

    def count_possessive_pronouns(tokens):
        possessive_pronouns = [token for token, pos in tokens if pos == 'POS' or pos == 'PRP$']
        return len(possessive_pronouns)

    filename = "ml/spam_mail_big.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            df = pickle.load(f)
    else:
        # Add the features
        df['tokens'] = df['text'].apply(word_tokenize)
        df['char_length'] = df['text'].apply(len)
        df['token_length'] = df['tokens'].apply(len)
        df['pos_tags'] = df['tokens'].apply(pos_tag)
        df['num_nouns'] = df['pos_tags'].apply(count_nouns)
        df['num_stopwords'] = df['tokens'].apply(count_stopwords)
        df['avg_token_length'] = df['tokens'].apply(avg_token_length)
        df['num_special_chars'] = df['text'].apply(count_special_chars)
        df['num_uppercase_words'] = df['tokens'].apply(count_uppercase_words)
        df['num_adverbs'] = df['pos_tags'].apply(count_adverbs)
        df['num_personal_pronouns'] = df['pos_tags'].apply(count_personal_pronouns)
        df['num_possessive_pronouns'] = df['pos_tags'].apply(count_possessive_pronouns)
        df['num_capital_letters'] = df['text'].apply(count_capital_letters)

        with open(filename, "wb") as f:
            pickle.dump(df, f)
            
    export_to_pickle(df.head(), 'spam_dataframe_features_head.pkl')
    return df

@st.cache_data #for caching the data in streamlit
def spam_mail_tsne(df):
    # Define the features we want to use
    features = ['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters']

    filename = "ml/spam_mail_tsne_big.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            spam_mail = pickle.load(f)
    else:
        # Extract the features we want to use for visualization
        X = df[features]
        tsne = TSNE(n_components=3, random_state=42)
        spam_mail = tsne.fit_transform(X)

        with open(filename, "wb") as f:
            pickle.dump(spam_mail, f)

    return spam_mail
    
def create_3d_visualization(X_tsne, df):
    # Create a Trace object for each class label
    classes = df['target'].unique() # 0 and 1
    traces = []
    for c in classes:
        if c == 1:
            name = 'Spam'
        else:
            name = 'Ham'
        mask = df['target'] == c
        color = 'red' if c == 1 else 'blue'
        trace = go.Scatter3d(x=X_tsne[mask, 0], y=X_tsne[mask, 1], z=X_tsne[mask, 2], mode='markers',
                             marker=dict(color=df['target'], colorscale=[[0, color], [1, color]], size=5, opacity=0.8),
                             name=str(name))
        traces.append(trace)

    # Add the traces to the data list of the Figure object
    fig = go.Figure(data=traces)

    # Set the layout parameters, including the legend
    fig.update_layout(scene=dict(xaxis_title='t-SNE Dimension 1', yaxis_title='t-SNE Dimension 2',
                                  zaxis_title='t-SNE Dimension 3'),
                      title='t-SNE Visualization of Email Features', showlegend=True)
    
    export_to_pickle(fig, "3d_visualization_dataframe.pkl")
    return fig

def visualize_onclasssvm(df, y_pred):
    X_tsne = spam_mail_tsne(df)

    df['predicted'] = y_pred
    conditions = [(df['target'] == 0) & (df['predicted'] == 0),
    (df['target'] == 0) & (df['predicted'] == 1),
    (df['target'] == 1) & (df['predicted'] == 1),
    (df['target'] == 1) & (df['predicted'] == 0)
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
            name = 'Non-Spam Emails Correctly Classified'
        elif c == 1:
            color = 'red'
            name = 'Non-Spam Emails Incorrectly Classified'
        elif c == 2:
            color = 'green'
            name = 'Spam Emails Correctly Classified'
        else:
            color = 'orange'
            name = 'Spam Emails Incorrectly Classified'

        
        trace = go.Scatter3d(x=X_tsne[mask, 0], y=X_tsne[mask, 1], z=X_tsne[mask, 2], mode='markers',
                             marker=dict(color=color, colorscale=[[0, color], [1, color]], size=5, opacity=0.8),
                             name=str(name))
        traces.append(trace)

    # Add the traces to the data list of the Figure object
    fig = go.Figure(data=traces)

    # Set the layout parameters, including the legend
    fig.update_layout(scene=dict(xaxis_title='t-SNE Dimension 1', yaxis_title='t-SNE Dimension 2',
                                  zaxis_title='t-SNE Dimension 3'),
                      title='t-SNE Visualization of the Predicted Email Features', showlegend=True)
    
    return fig

def create_oneclass_svm_predict(df, kernel, nu, gamma, degree=3, outlier_fraction=False, gamma_scale=False):
    # separate the features and target columns
    X = df[['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters']]
    y = df['target']

    X = scale_data(X)

    # create the OneClassSVM model
    if gamma_scale:
        gamma = 'scale'
    
    if outlier_fraction:
        nu = len(df[df['target']==1])/float(len(df[df['target']==0]))
        print(f'Outlier Fraction: {outlier_fraction}')

    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)

    clf.fit(X)

    # evaluate the model using cross-validation
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision_macro': make_scorer(precision_score, average='macro',zero_division=1),
               'recall_macro': make_scorer(recall_score, average='macro', zero_division=1),
               'f1_macro': make_scorer(f1_score, average='macro', zero_division=1),
               'roc_auc': make_scorer(roc_auc_score, average='macro')}
    
    cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)

    # return the average scores from the cross-validation
    y_pred = clf.predict(X) # use the model to detect outliers (spam emails)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # the result
    accuracy = accuracy_score(y,y_pred)
    precision = precision_score(y,y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    auc_roc = roc_auc_score(y,y_pred)

    # calculate the mean scores across all folds
    cv_accuracy = cv_results['test_accuracy'].mean()
    cv_precision = cv_results['test_precision_macro'].mean()
    cv_recall = cv_results['test_recall_macro'].mean()
    cv_f1 = cv_results['test_f1_macro'].mean()
    cv_auc_roc = cv_results['test_roc_auc'].mean()

    return y_pred, [cv_accuracy, cv_precision, cv_recall, cv_f1, cv_auc_roc], [accuracy, precision, recall, f1, auc_roc]

def export_and_visualize_oneclasssvm(df, kernel, nu, gamma, degree, outlier_fraction, gamma_scale):
    y_pred, cv_res, res = create_oneclass_svm_predict(df, kernel, nu, gamma, degree, outlier_fraction, gamma_scale)
    fig = visualize_onclasssvm(df, y_pred)
    cv_values = {'Accuracy': cv_res[0], 'Precision': cv_res[1], 'Recall': cv_res[2], 'F1 Score': cv_res[3], 'AUC ROC': cv_res[4]}
    values = {'Accuracy': res[0], 'Precision': res[1], 'Recall': res[2], 'F1 Score': res[3], 'AUC ROC': res[4]}
    settings = {'kernel': kernel, 'nu': nu, 'gamma': gamma, 'degree': degree}

    export_to_pickle(fig, f"visualize_onclasssvm_{kernel}.pkl")
    export_to_pickle(cv_values, f"cv_values_onclasssvm_{kernel}.pkl")
    export_to_pickle(values, f"values_onclasssvm_{kernel}.pkl")
    export_to_pickle(settings, f"settings_onclasssvm_{kernel}.pkl")

    return fig, values, cv_values
    
def create_best_model(df):
    # define the values to try for each hyperparameter
    kernels = ['rbf']
    nus = np.arange(0.10, 1.0, 0.10)
    gammas = np.arange(0.5, 50.0, 2)

    # initialize variables to store the best hyperparameters and metrics
    best_params = {}
    best_metrics = {'F1 Score': 0.0}

    # loop through all combinations of hyperparameters
    for kernel, nu, gamma in itertools.product(kernels, nus, gammas):
        # fit the OneClassSVM model and calculate metrics
        y_pred, cv_res, res = create_oneclass_svm_predict(df, kernel, nu, gamma)
        print(f"kernel: {kernel}, nu: {nu}, gamma: {gamma}, accuracy: {cv_res[0]}, precision: {cv_res[1]}, recall: {cv_res[2]}, f1: {cv_res[3]}, auc_roc: {cv_res[4]}")
        
        # check if this set of hyperparameters produced the best F1 Score so far
        if cv_res[3] > best_metrics['F1 Score']:
            best_params = {'kernel': kernel, 'nu': nu, 'gamma': gamma}
            best_metrics = {'accuracy': cv_res[0], 'precision': cv_res[1], 'recall': cv_res[2], 'F1 Score': cv_res[3], 'AUC ROC': cv_res[4]}
    
    return best_metrics, best_params
