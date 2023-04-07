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

# Download the NLTK Data
nltk.download('punkt')

@st.cache_data #for caching the data in streamlit
def get_dataset():
     # Load the data
    # https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset?resource=download
    return pd.read_csv('spam_assassin.csv')

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

    filename = "spam_mail_big.pkl"
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

    return df

@st.cache_data #for caching the data in streamlit
def spam_mail_tsne(df):
    # Define the features we want to use
    features = ['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters']

    filename = "spam_mail_tsne_big.pkl"
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
    return fig




def visualize_3d_onclasssvm_linear(X, y_pred, clf):
    # create the 3D scatter plot
    trace1 = go.Scatter3d(
        x=X[y_pred == 0]['char_length'],
        y=X[y_pred == 0]['num_nouns'],
        z=X[y_pred == 0]['num_adverbs'],
        mode='markers',
        name='Non-Spam Emails',
        marker=dict(
            color='blue',
            size=5,
            opacity=0.8
        )
    )

    trace2 = go.Scatter3d(
        x=X[y_pred == 1]['char_length'],
        y=X[y_pred == 1]['num_nouns'],
        z=X[y_pred == 1]['num_adverbs'],
        mode='markers',
        name='Spam Emails',
        marker=dict(
            color='green',
            size=5,
            opacity=0.8
        )
    )

    trace3 = go.Surface(
        x=X['char_length'],
        y=X['num_nouns'],
        z=(-clf.intercept_[0] - clf.coef_[0][0] * X['char_length'] - clf.coef_[0][1] * X['num_nouns']) / clf.coef_[0][2],
        name='Hyperplane',
        showscale=False,
        opacity=0.9,
        colorscale='Blues'
    )

    layout = go.Layout(
        title='OneClassSVM Email Classifier',
        scene=dict(
            xaxis=dict(title='Character Length'),
            yaxis=dict(title='Number of Nouns'),
            zaxis=dict(title='Number of Adverbs')
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    
    return fig

import plotly.graph_objs as go
import numpy as np

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
                      title='t-SNE Visualization of Email Features', showlegend=True)
    return fig



@st.cache_resource
def create_oneclass_svm_predict(df, kernel, nu, gamma, degree):
    # separate the features and target columns
    X = df[['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters']]
    y = df['target']

    # create and fit the OneClassSVM model
    clf = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)
    clf.fit(X[y == 0])  # fit the model on non-spam emails
    y_pred = clf.predict(X) # use the model to detect outliers (spam emails)

    # 1 represents outliers (spam emails) in OneClassSVM
    # replace 1 with 0 to get the target values as 0 for non-spam and 1 for spam
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # calculate the accuracy score
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred)
    
    fig = visualize_onclasssvm(df, y_pred)

    return fig, {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'AUC ROC': auc_roc}

