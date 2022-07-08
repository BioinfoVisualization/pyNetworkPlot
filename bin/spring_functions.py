""" Functions that generate a set of edge weights for a given igraph.Graph object

Force-based network visualization algorithms, like Fruchterman Reingold or DrL, model a network as a set of repelling objects (nodes) connected by springs that pull them together (edges). The implementation of these algorithms in igraph allows each edge to have a unique spring constant K (or weight) to fine-tune the visualization. The following functions generate different distributions of spring weights depending on the graph topology, assigning a unique K to edges in the same connected component (or cluster) of the graph. All functions work on a previously initialized igraph.Graph object, whose nodes have been labelled according to the connected component of the graph to which they belong and the number of nodes in each connected component. These variables are stored in the node data frame of the graph as 'plot_cluster_reduced' and 'freq_cluster', respectively.
    * logspaced_weights
    * scaled_log_weights
    * inverse_cluster_size_weights
    * prop_log_weights

Author: Juan Sebastian Diaz Boada
12/12/21
"""
import numpy as np
import pandas as pd
import igraph as ig
#----------------------------------------------------------------------------------------------------
def logspaced_weights(g):
    """ Creates an increasing logarithmic difference between the different clusters' weights.
    
        Calculates a weight value for each cluster, giving the biggest cluster 10^-7 and the 
        single nodes 10^-1. The change in the weights values increases logarithmically between
        clusters, so clusters with more than 2 nodes tend to have very low weights, and small 
        clusters have weights close to 10^-1. Useful when the majority of nodes belog to a 
        cluster. May not give optimal results when there are many clusters with
        the same low number of nodes.
        
        Parameters
        -----------
        g : igraph.Graph
            Graph initialized with node atribute 'plot_cluster_reduced'.
        
        Returns
        -------
        np.array
            Array of length equal to the number of edges in g.
            
    """
    # Spring constants (Ks) for edges of each cluster
    Ks = np.logspace(-7,-1,num=len(np.unique(g.vs['plot_cluster'])))
    # Edge assignment
    edges_array = np.array(g.get_edgelist())
    node_1 = np.array(g.vs['cluster'])[edges_array[:,0]]
    node_2 = np.array(g.vs['cluster'])[edges_array[:,1]]
    weights = Ks[np.minimum(node_1,node_2)]
    return weights
#----------------------------------------------------------------------------------------------------
def scaled_log_weights(g):
    """ Creates a decreasing logarithmic difference between the different clusters' weights.
        
        Calculates a weight value for each cluster, giving the biggest cluster 10^-7 and the 
        single nodes 10^-1. The change in the weights values decreases logarithmically between
        clusters, so the biggest cluster will have a significally smaller weight than the second
        biggest. Useful when the majority of nodes are not connected or in small clusters, and 
        there are few big clusters. May not give optimal results when the biggest clusters have a
        similar number of nodes.
        
        Parameters
        -----------
        g : igraph.Graph
            Graph initialized with node atribute 'plot_cluster_reduced'.
        
        Returns
        -------
        np.array
            Array of length equal to the number of edges in g.
    """
    x = np.log10(np.linspace(0.01,1,num=len(np.unique(g.vs['plot_cluster']))))
    x = (x - np.min(x))
    Ks = x/(100*np.max(x))+1e-7
    # Edge assignment
    edges_array = np.array(g.get_edgelist())
    node_1 = np.array(g.vs['cluster'])[edges_array[:,0]]
    node_2 = np.array(g.vs['cluster'])[edges_array[:,1]]
    weights = Ks[np.minimum(node_1,node_2)]
    return weights
#----------------------------------------------------------------------------------------------------
def inverse_cluster_size_weights(g):
    """ Creates weights equal to the normalized inverse of the number of nodes in the clusters.
    
        Useful when several clusters have the same number of nodes. May not give optimal results
        when the mjority of the nodes are not connected.
        
        Parameters
        -----------
        g : igraph.Graph
            Graph initialized with node atribute 'plot_cluster_reduced'.
        
        Returns
        -------
        np.array
            Array of length equal to the number of edges in g.
    """
    Ks = 1/np.array(g.vs['freq'])
    norm = np.max(Ks)
    Ks = Ks/norm
    # Edge assignment
    edges_array = np.array(g.get_edgelist())
    node_1 = Ks[edges_array[:,0]]
    node_2 = Ks[edges_array[:,1]]
    weights = np.minimum(node_1,node_2)
    return weights
#----------------------------------------------------------------------------------------------------
def prop_log_weights(g):
    """ Weights are inverse of product of number of clusters and number of nodes in cluster.
    
        Calculates the weight of each edge according to (1/(L*N_c)), where L is the total number
        of connected components in the graph (taking the unconnected nodes as one connected 
        component), and N_c is the number of nodes in the cluster where the edge is. Useful in 
        the majority of cases (nodes<1000) and is the default function.
        
        Parameters
        -----------
        g : igraph.Graph
            Graph initialized with node atribute 'plot_cluster_reduced'. 
        
        Returns
        -------
        np.array
            Array of length equal to the number of edges in g.
    """
    # Turn to df to take advantage the ordering of pd.unique instead of np.unique
    df = g.get_vertex_dataframe()
    clusters = df.loc[:,'cluster'].unique() # Cluster numbers 1..len(clusters)-1
    L = len(clusters)
    F = np.zeros([L],dtype=int)
    for c in clusters:
        F[c] = df.loc[df['cluster']==c,'freq'].values[0]
    Ks = 1/(L*F)
    # Edge assignment
    edges_array = np.array(g.get_edgelist())
    node_1 = np.array(g.vs['cluster'])[edges_array[:,0]]
    node_2 = np.array(g.vs['cluster'])[edges_array[:,1]]
    weights = Ks[np.minimum(node_1,node_2)]
    return weights