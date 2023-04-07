import streamlit as st
from spam_mail import spam_mail_features, get_dataset, spam_mail_tsne, create_3d_visualization, create_oneclass_svm_predict, visualize_onclasssvm
import inspect

def spam_mail_local(tab, kernel, nu, gamma, degree, score):

    ### Load the mail dataset
    df = get_dataset()
    tab.dataframe(df.head(), use_container_width=True)

    ## example of ham mail
    with tab.expander('Example of Ham Mail', expanded=False):
        st.write(df['text'][0])

    ## example of spam mail
    with tab.expander('Example of Spam Mail', expanded=False):
        st.write(df['text'][1])

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(get_dataset)
        st.code(code)

    ### Feature Extraction
    tab.subheader('Feature Extraction')
    tab.write('We use the nltk library to create the features for the dataset. We create the following features:')
    tab.write(['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters'])

    df_mail_w_features = spam_mail_features(df)
    tab.dataframe(df_mail_w_features.head(), use_container_width=True)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(spam_mail_features)
        st.code(code)

    ### Create 3d Visualization with t-SNE
    tab.subheader('Create 3d Visualization with t-SNE')
    X = spam_mail_tsne(df_mail_w_features)
    fig = create_3d_visualization(X, df_mail_w_features)
    tab.plotly_chart(fig)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(create_3d_visualization)
        st.code(code)

    ### One Class SVM
    tab.subheader('One Class SVM Prediction')

    fig, result = create_oneclass_svm_predict(df_mail_w_features, kernel, nu, gamma, degree)
    
    col1, col2 = tab.columns(2)
    col3, col4 = tab.columns(2)
    col5, col6 = tab.columns(2)

    col1.write('Kernel: ' + kernel)
    col2.write('Nu: ' + str(nu))
    col3.write('Gamma: ' + str(gamma))
    col4.write('Degree: ' + str(degree))
    col5.write('Results:')
    col6.write(result)

    tab.plotly_chart(fig)


    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(create_oneclass_svm_predict)
        st.code(code)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(visualize_onclasssvm)
        st.code(code)
