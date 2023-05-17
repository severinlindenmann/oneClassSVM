import streamlit as st
import spam_mail as spm
import credit_card as cc
import inspect

def spam_mail_local(tab, kernel, nu, gamma, degree, score, outlier_fraction, gamma_scale):



    ### Load the mail dataset
    df = spm.get_dataset()
    tab.dataframe(df.head(), use_container_width=True)

    if gamma_scale:
        gamma = 'scale'
    
    if outlier_fraction:
        nu = len(df[df['target']==1])/float(len(df[df['target']==0]))
        print(f'Outlier Fraction: {nu}')

    ## example of ham mail
    with tab.expander('Example of Ham Mail', expanded=False):
        st.write(df['text'][0])

    ## example of spam mail
    with tab.expander('Example of Spam Mail', expanded=False):
        st.write(df['text'][1])

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(spm.get_dataset)
        st.code(code)

    ### Feature Extraction
    tab.subheader('Feature Extraction')
    tab.write('We use the nltk library to create the features for the dataset. We create the following features:')
    tab.write(['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters'])

    df_mail_w_features = spm.spam_mail_features(df)
    tab.dataframe(df_mail_w_features.head(), use_container_width=True)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(spm.spam_mail_features)
        st.code(code)

    ### Create 3d Visualization with t-SNE
    tab.subheader('Create 3d Visualization with t-SNE')
    tab.write('We use the t-SNE algorithm to create a 3d visualization of the dataset. The visualization shows the spam and ham mails in a 3d space. The visualization shows that the spam mails and ham mails show some patterns. The ham mail create a spiral and the spam mails are more outside of the spiral.')
    X = spm.spam_mail_tsne(df_mail_w_features)
    fig = spm.create_3d_visualization(X, df_mail_w_features)
    tab.plotly_chart(fig)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(spm.spam_mail_tsne)
        st.code(code)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(spm.create_3d_visualization)
        st.code(code)

    ### One Class SVM
    tab.subheader('One Class SVM Prediction')
    tab.write('We use the One Class SVM algorithm to predict the spam mails. We use the following parameters settings:')
    tab.subheader('Settings')

    fig, result, cv_result = spm.export_and_visualize_oneclasssvm(df_mail_w_features, kernel, nu, gamma, degree, outlier_fraction, gamma_scale)
    
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

    # tab.subheader('Cross Validation Results')
    # tab.write('With Cross Validation we get a much lower result:')
    # col9, col10 = tab.columns(2)
    # col11, col12 = tab.columns(2)

    # col9.metric('Accuracy', cv_result['Accuracy'])
    # col10.metric('Precision', cv_result['Precision'])
    # col11.metric('Recall', cv_result['Recall'])
    # col12.metric('F1 Score', cv_result['F1 Score'])

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(spm.create_oneclass_svm_predict)
        st.code(code)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(spm.visualize_onclasssvm)
        st.code(code)


    ### Evaluate the best Model
    evaluate = False
    if evaluate:
        tab.title('Evaulate the best Model')
        tab.write('We use the best model to predict the spam mails.')
        result = spm.create_best_model(df_mail_w_features)
        tab.write(result)


def credit_card_local(tab, kernel, nu, gamma, degree, score, outlier_fraction, gamma_scale):

    ### Load the cc dataset
    df = cc.get_dataset()
    tab.dataframe(df.head(), use_container_width=True)


    if gamma_scale:
        gamma = 'scale'
    
    if outlier_fraction:
        nu = len(df[df['Class']==0])/float(len(df[df['Class']==1]))
        print(f'Outlier Fraction: {outlier_fraction}')

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(cc.get_dataset)
        st.code(code)

    ### Create 3d Visualization with t-SNE
    tab.subheader('Create 3d Visualization with t-SNE')
    tab.write('We use the t-SNE algorithm to create a 3d visualization of the dataset. The visualization shows the spam and ham mails in a 3d space. The visualization shows that the spam mails and ham mails show some patterns. The ham mail create a spiral and the spam mails are more outside of the spiral.')
    X = cc.credit_card_tsne(df)
    fig = cc.create_3d_visualization(X, df)
    tab.plotly_chart(fig)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(cc.credit_card_tsne)
        st.code(code)

    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(cc.create_3d_visualization)
        st.code(code)

    ### One Class SVM
    tab.subheader('One Class SVM Prediction')
    tab.write('We use the One Class SVM algorithm to predict the cc frauds. We use the following parameters settings:')
    tab.subheader('Settings')

    fig, result, cv_result = cc.export_and_visualize_oneclasssvm(df, kernel, nu, gamma, degree, outlier_fraction, gamma_scale)
    
    ## show the code
    with tab.expander('Code', expanded=False):
        code = inspect.getsource(cc.export_and_visualize_oneclasssvm)
        st.code(code)

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

    # tab.subheader('Cross Validation Results')
    # tab.write('With Cross Validation we get a much lower result:')
    # col9, col10 = tab.columns(2)
    # col11, col12 = tab.columns(2)

    # col9.metric('Accuracy', cv_result['Accuracy'])
    # col10.metric('Precision', cv_result['Precision'])
    # col11.metric('Recall', cv_result['Recall'])
    # col12.metric('F1 Score', cv_result['F1 Score'])