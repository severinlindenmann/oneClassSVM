import streamlit as st
from oneclass_svm_sample import generate_example_data, create_oneclasssvm_demo, create_oneclasssvm_2d_countour_demo
from oneclass_svm_local import spam_mail_local
import inspect
from dotenv import load_dotenv
import os
load_dotenv()

### check if run locally or in the cloud, the cloud can't handle the large dataset (performance issue)
LOCAL = os.getenv('LOCAL')

### Intro
st.title('Outlier Detection')
st.subheader('One Class SVM')
st.write('One-class SVM is an unsupervised algorithm that learns a decision function for novelty detection: classifying new data as similar or different to the training set.')

### Sidebar - Collects user input features into dataframe
st.sidebar.write('This dashboard is used to detect outliers in the dataset')
st.sidebar.title('Hyperparameters')

kernel = st.sidebar.selectbox('Kernel', ('rbf', 'linear', 'poly', 'sigmoid'))
nu = st.sidebar.slider('NU', 0.05, 0.95, 0.05)
gamma = st.sidebar.slider('GAMMA', 0.5, 50.0, 0.5)
score = st.sidebar.slider('SCORE', 0.1, 1.0, 0.5, 0.1)

degree = 0 
if kernel == 'poly': # only for poly
    degree = st.sidebar.slider('Degree', 0, 10, 1)

st.sidebar.title('Explore a Dataset')
datasets = st.sidebar.selectbox('Dataset', ('Spam Mail', 'Spam Mail'))

### Example Data
tab1, tab2 = st.tabs(['Example Data', datasets])
tab1.subheader('Example Data')
tab1.write('The dataset is generated using the make_classification function from the sklearn.datasets module. The dataset contains 200 data points with 2 features. The data points are divided into 2 classes.')
tab1.write('The two classes should demostrate the normal and outlier data points.')

### generate example data
x, _, fig = generate_example_data()

### plot the example data
tab1.pyplot(fig)

## show the code
with tab1.expander('Code', expanded=False):
    code = inspect.getsource(generate_example_data)
    st.code(code)

### One Class SVM Demo
fig, svm = create_oneclasssvm_demo(x,_ , kernel, nu, gamma, degree, score)
tab1.write(f'The outliers are detected using the OneClassSVM function from the sklearn.svm module. The outliers are detect using the score_samples function. The outliers are detected using the threshold value of {score}.')
tab1.write('the red dots are the outliers')
tab1.pyplot(fig)
  
## show the code
with tab1.expander('Code', expanded=False):
    code = inspect.getsource(create_oneclasssvm_demo)
    st.code(code)

fig = create_oneclasssvm_2d_countour_demo(svm, x, _)
tab1.write('The decision contour is plotted using the predict function from the OneClassSVM module.')
tab1.pyplot(fig)

## show the code
with tab1.expander('Code', expanded=False):
    code = inspect.getsource(create_oneclasssvm_2d_countour_demo)
    st.code(code)

### Dataset
## Check if the dataset is selected and load the dataset
if datasets == 'Spam Mail':
    tab2.info("Click Render & Update in the sidebar left to load the dataset and train the model")
    

    tab2.subheader('Spam Mail')
    tab2.write('The dataset is loaded using the read_csv function from the pandas module. The dataset contains 5329 emails. The emails are divided into 2 classes: spam and ham.')
    tab2.write('3900 no spam (ham) and 1896 is spam.')
    tab2.write('Source: https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset?resource=download')
    
    update = st.sidebar.button('Render & Update')
    if update:
        if LOCAL == 'FALSE':
            tab2.info('The dataset is too large to be loaded and process live in the cloud, we will show you the preprocessed dataset and model')

        if LOCAL == 'TRUE':
            with st.spinner('Wait for it...'):
                tab2.info("It will take a while to load the dataset and train the model, please have patience")
                spam_mail_local(tab2, kernel, nu, gamma, degree, score)

print('SUCCESSFULLY RUN')