    tab2.subheader('Data Table')
    @st.cache_data
    def load_pickle_df(tt):   
        with open(f'{tt}.pkl', 'rb') as pickle_file:
            return pickle.load(pickle_file)
    
    df = load_pickle_df('spam_mail')

    tab2.subheader('3d Plot of Features')
    tab2.write('We use the t-SNE algorithm to project the data onto a 3D space. The plot shows that the spam emails are more spread out than the non-spam emails.')
    tab2.write('0: Non-Spam')
    tab2.write('1: Spam')

    # Extract the features we want to use for visualization
    yy = load_pickle_df('spam_mail_target')
    features = ['char_length', 'token_length', 'num_nouns', 'num_stopwords', 'avg_token_length', 'num_special_chars', 'num_uppercase_words', 'num_adverbs', 'num_personal_pronouns', 'num_possessive_pronouns', 'num_capital_letters']

    @st.cache_resource
    def load_tsne():
        with open('TSNE_x.pkl', 'rb') as pickle_file:
            return pickle.load(pickle_file)

    X_tsne = load_tsne()

    # Create a Trace object for each class label
    classes = yy.unique()
    traces = []
    for c in classes:
        mask = yy == c
        color = 'red' if c == 1 else 'blue'
        trace = go.Scatter3d(x=X_tsne[mask, 0], y=X_tsne[mask, 1], z=X_tsne[mask, 2], mode='markers',
                             marker=dict(color=yy, colorscale=[[0, color], [1, color]], size=5, opacity=0.8),
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

    with tab2.expander('Code for visualization'):
        st.code("""
    # Create a Trace object for each class label
    classes = yy.unique()
    traces = []
    for c in classes:
        mask = yy == c
        color = 'red' if c == 1 else 'blue'
        trace = go.Scatter3d(x=X_tsne[mask, 0], y=X_tsne[mask, 1], z=X_tsne[mask, 2], mode='markers',
                             marker=dict(color=yy, colorscale=[[0, color], [1, color]], size=5, opacity=0.8),
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

""")
