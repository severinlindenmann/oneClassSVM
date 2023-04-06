import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_classification
from numpy import quantile, where
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objs as go
import urllib.request

st.title('Outlier Detection')
st.subheader('One Class SVM')
st.write('One-class SVM is an unsupervised algorithm that learns a decision function for novelty detection: classifying new data as similar or different to the training set.')

st.sidebar.write('This dashboard is used to detect outliers in the dataset')
st.sidebar.title('Hyperparameters')

kernel = st.sidebar.selectbox('Kernel', ('rbf', 'linear', 'poly', 'sigmoid'))
nu = st.sidebar.slider('NU', 0.05, 0.95, 0.05)
gamma = st.sidebar.slider('GAMMA', 0.5, 50.0, 0.5)

degree = 0
if kernel == 'poly':
    degree = st.sidebar.slider('Degree', 0, 10, 1)


st.sidebar.title('Explore a Dataset')
datasets = st.sidebar.selectbox('Dataset', ('Spam Mail', 'Spam Mail'))

tab1, tab2 = st.tabs(['Example Data', datasets])
tab1.subheader('Example Data')


# Generate the example data
x, _ = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=13)

# Create a scatter plot of the data points
fig, ax = plt.subplots()
scatter = ax.scatter(x[:, 0], x[:, 1], c=_)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter plot of example data')
plt.colorbar(scatter)

tab1.write('The dataset is generated using the make_classification function from the sklearn.datasets module. The dataset contains 200 data points with 2 features. The data points are divided into 2 classes.')
tab1.pyplot(fig)
svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)

pred = svm.fit_predict(x)
scores = svm.score_samples(x)

thresh = quantile(scores, 0.5)
index = where(scores<=thresh)
values = x[index]

fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:,1], c=_)
ax.scatter(values[:,0], values[:,1], color='r')

tab1.write('The outliers are detected using the OneClassSVM function from the sklearn.svm module. The outliers are detect using the score_samples function. The outliers are detected using the threshold value of 0.05.')
tab1.pyplot(fig)

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
                     

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the scatter plot of the data points
fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:,1])

# Plot the decision contour
ax.contourf(xx, yy, Z, alpha=0.2)

tab1.write('The decision contour is plotted using the predict function from the OneClassSVM module.')
tab1.pyplot(fig)


if datasets == 'Spam Mail':
    tab2.subheader('Spam Mail')
    tab2.write('The dataset is loaded using the read_csv function from the pandas module. The dataset contains 5329 emails, 3900 no spam and 1896 is spam.')
    tab2.write('Source: https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset?resource=download')
    tab2.subheader('Data Feature Creation')
    tab2.write('We use the nltk library to create the features for the dataset. We create the following features:')
    tab2.write(['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters'])

    with tab2.expander('Code'):
        st.code("""
# Import the Libraries
import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download the NLTK Data
nltk.download('punkt')

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

def count_adjectives(tokens):
    adjectives = [token for token, pos in tokens if pos.startswith('JJ')]
    return len(adjectives)

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

# Load the data
df = pd.read_csv('spam_assassin.csv')

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

df.to_pickle('spam_mail.pkl')
# https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset?resource=download
""")
    tab2.subheader('Data Table')

    @st.cache_data
    def load_pickle_df():   
        url = 'https://s3.severin.io/ml%2Fspam_mail.pkl'
        with urllib.request.urlopen(url) as f:
            return pickle.load(f)
    
    df = load_pickle_df()

    # tab2.pyplot(fig)
    tab2.dataframe(df)

    tab2.subheader('2d Plot of Features')
    tab2.write('We use the t-SNE algorithm to project the data onto a 3D space. The plot shows that the spam emails are more spread out than the non-spam emails.')
    tab2.write('0: Non-Spam')
    tab2.write('1: Spam')

    features = ['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters']
    # Extract the features we want to use for visualization
    X = df[features]
    y = df['target']

    @st.cache_resource
    def load_tsne(X):
        url = 'https://s3.severin.io/ml%2FX_tsne.npy'
        with urllib.request.urlopen(url) as f:
            return np.load(f)
        # # Perform t-SNE to project the data onto a 3D space
        # tsne = TSNE(n_components=3, random_state=42)
        # return tsne.fit_transform(X)

    # Create a Trace object for each class label
    classes = df['target'].unique()
    traces = []
    for c in classes:
        mask = df['target'] == c
        color = 'red' if c == 1 else 'blue'
        trace = go.Scatter3d(x=X_tsne[mask, 0], y=X_tsne[mask, 1], z=X_tsne[mask, 2], mode='markers',
                             marker=dict(color=y[mask], colorscale=[[0, color], [1, color]], size=5, opacity=0.8),
                             name=str(c))
        traces.append(trace)

    # Add the traces to the data list of the Figure object
    fig = go.Figure(data=traces)

    # Set the layout parameters, including the legend
    fig.update_layout(scene=dict(xaxis_title='t-SNE Dimension 1', yaxis_title='t-SNE Dimension 2',
                                  zaxis_title='t-SNE Dimension 3'),
                      title='t-SNE Visualization of Email Features', showlegend=True)



    # Display plot in Streamlit
    tab2.plotly_chart(fig)