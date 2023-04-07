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
    tab.write('We use the t-SNE algorithm to create a 3d visualization of the dataset. The visualization shows the spam and ham mails in a 3d space. The visualization shows that the spam mails and ham mails show some patterns. The ham mail create a spiral and the spam mails are more outside of the spiral.')
    X = spam_mail_tsne(df_mail_w_features)
    fig = create_3d_visualization(X, df_mail_w_features)
    tab.plotly_chart(fig)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(create_3d_visualization)
        st.code(code)

    ### One Class SVM
    tab.subheader('One Class SVM Prediction')
    tab.write('We use the One Class SVM algorithm to predict the spam mails. We use the following parameters settings:')
    tab.subheader('Settings')

    fig, result = create_oneclass_svm_predict(df_mail_w_features, kernel, nu, gamma, degree)
    
    col1, col2 = tab.columns(2)
    col3, col4 = tab.columns(2)

    col1.metric('Kernel', kernel)
    col2.metric('Nu', nu)
    col3.metric('Gamma', gamma)
    col4.metric('Degree', degree)

    tab.subheader('Results')
    tab.write('We got with this kernel the following results:')
    col5, col6 = tab.columns(2)
    col7, col8 = tab.columns(2)

    col5.metric('Accuracy', result['Accuracy'])
    col6.metric('Precision', result['Precision'])
    col7.metric('Recall', result['Recall'])
    col8.metric('F1 Score', result['F1 Score'])

    tab.write('Like before we create a 3d visualization of the dataset, but know we match the colors if the mail is a ham or spam and if it is predicted correctly or not.')
    tab.write('The rbf kernel shows the best results. The sigmoid kernel shows the worst results, but that could also be our fault because we did not tune the parameters good enough for sigmoid.')
    
    tab.plotly_chart(fig)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(create_oneclass_svm_predict)
        st.code(code)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(visualize_onclasssvm)
        st.code(code)
