def mds_plot(cluster_centers):
    
    #INPUTS:
    # CLUSTER CENTERS DICTIONARY
    
    #OUTPUTS:
    # LIST OF INTERACTIVE PLOTLY MDS FIGURES
    
    # IMPORTS
    import plotly.graph_objects as go
    from sklearn.manifold import MDS
    import plotly.express as px
    from sklearn.preprocessing import MinMaxScaler
    from collections import Counter
    
    fig = go.Figure()
    fig_list = []
    for _key in cluster_centers.keys():
        if _key:
            cluster_centers_df = pd.DataFrame(cluster_centers[_key][0])
            # MDS
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(cluster_centers_df)
            mds = MDS(2,random_state=0)
            X_2d = mds.fit_transform(X_scaled)
            mds_df = pd.DataFrame(data = X_2d, columns = range(1,len(X_2d[0])+1))
            mds_df.columns = ['X'+str(col) for col in mds_df.columns]
            mds_df.insert(0, 'SegId', mds_df.index +1)

            Counter(cluster_labels[type_][0])
            mds_size = pd.DataFrame()
            mds_size['SegId'] = Counter(cluster_labels[type_][0]).keys()
            mds_size['Size'] = Counter(cluster_labels[type_][0]).values()
            mds_size.sort_values(by=['SegId'], inplace = True)
            mds_df['Size'] = mds_size['Size']

            # Plot MDS
            mds_df['Cluster'] = mds_df['SegId'].astype(str)
            mds_df['Cluster'] = 'Cluster ' + mds_df['Cluster']
            fig = px.scatter(mds_df, x="X1", y="X2", color='Cluster',
                             title= _key + ' ' +str(len(mds_df))+"Segments",
                             size="Size", hover_name="SegId", text="Cluster")
            fig.update_traces(showlegend=False,textposition='top center')
            fig_list.append(fig)
    return fig_list


fig_list = mds_plot(cluster_centers)
for fig in fig_list:
    fig.show()