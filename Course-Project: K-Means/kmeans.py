'''
Problem Definition: Apply K-Means algorithm to cluster nodes from two graph
datasets.

Authors: Demarco Guajardo and Richard Harris
'''

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score

def load_data(file_path):
    ''' 
    Load the dataset from a text file.
    '''
    data = pd.read_csv(
        file_path, 
        sep='\t',
        header=None,
        names=["FromNodeID", "ToNodeID"],
        dtype={'FromNodeID': str, 'ToNodeID': str}, # Treat node IDs as strings
        low_memory=False # Avoid memory issues with large datasets
    )
    return data

def reduce_dimensionality(data, n_components=2):
    '''
    Reduce the dimensionality of the data using PCA.
    '''
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def preprocess_directed_graph(data):
    '''
    Preprocess directed graph into a feature matrix
    Featurs in-degree and out-degree
    '''
    # Get unique nodes
    nodes = pd.unique(data[['FromNodeID', 'ToNodeID']].values.ravel())
    node_df = pd.DataFrame(nodes, columns=['NodeID'])
    
    # Calculate in-degree and out-degree
    in_degree = data['ToNodeID'].value_counts()
    out_degree = data['FromNodeID'].value_counts()

    # Merge in-degree and out-degree into a feature matrix
    node_df['InDegree'] = node_df['NodeID'].map(in_degree).fillna(0)
    node_df['OutDegree'] = node_df['NodeID'].map(out_degree).fillna(0)

    return node_df.set_index('NodeID')

def preprocess_undirected_graph(data):
    '''
    Preprocess undirected graph into a sparse adjacency matrix.
    '''
    # Get unique nodes
    nodes = pd.unique(data[['FromNodeID', 'ToNodeID']].values.ravel())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Map edges to indices
    row_indices = data['FromNodeID'].map(node_to_index)
    col_indices = data['ToNodeID'].map(node_to_index)

    # Create a sparse adjacency matrix
    adjacency_matrix = coo_matrix(
        (np.ones(len(data)), (row_indices, col_indices)),
        shape=(len(nodes), len(nodes))
    )

    return adjacency_matrix, nodes

def run_kmeans(data):
    '''
    Run K-Means clustering on the data.
    Evaluate results.
    '''

    # Convert sparse matrix to dense if necessary
    if isinstance(data, coo_matrix):
        data = data.toarray()

    # Give the user the option of choosing the number of clusters
    while True:
        try:
            k = int(input("Enter the number of clusters (k): "))
            if k <= 0:
                raise ValueError("Number of clusters must be greater than 0.")
            break
        except ValueError as e:
            print("Invalid input: {e}. Please enter a positive integer.")

    # Run K-Means
    kmeans = KMeans(n_clusters=k, random_state=88)
    labels = kmeans.fit_predict(data)

    # Evaluate Silhouette Coefficient and Davies-Bouldin Index
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)

    return labels, silhouette, davies_bouldin

def kmeans_results(data, visualize=True):
    '''
    Run K-Means and display results and any visualizations.
    Reports Silhouette Score and Davies-Bouldin Index (Effectiveness)
    Reports runtime (Efficiency)
    Visualizes clusters
    '''

    # Measure runtime
    start = time.time()
    labels, silhouette, davies_bouldin = run_kmeans(data)
    end = time.time()
    runtime = end - start

    # Print effectiveness and efficiency results
    print("\n~~~~~~~~~~ K-Means Clustering Results ~~~~~~~~~~")
    print(f"Silhouette Coefficient: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"Runtime: {runtime:.4f} seconds")

    # Visualize clusters
    if visualize and data.shape[1] == 2:
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', s=10)
        plt.title('K-Means Clustering Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label="Cluster")
        plt.show()
    elif not visualize:
        print("Visualization skipped.")
    else:
        print("Visualization skipped: Data is not 2D.")

def sample_data(data, sample_size=10000):
    '''
    Sample a subset of the data for faster processing.
    '''
    if sample_size > len(data):
        print(f"Sample size {sample_size} exceeds data length {len(data)}. Using full dataset.")
        return data
    return data.sample(n=sample_size, random_state=88)

# Main Function
if __name__ == "__main__":

    # Load and preprocess directed graph
    directed_filepath = "CA-GrQc.txt"
    directed_data = load_data(directed_filepath)
    directed_features = preprocess_directed_graph(directed_data)

    # Visualize results for directed graph
    print("\n~~~~~ Directed Graph K-Means Clustering (CA-GrQc.txt) ~~~~~")
    kmeans_results(directed_features, visualize=True)

    # Load and preprocess undirected graph
    undirected_filepath = "com-dblp.ungraph.txt"
    undirected_data = load_data(undirected_filepath)
    adjacency_matrix, nodes = preprocess_undirected_graph(undirected_data)

    # Sample undirected graph data
    #print("\nSampling undirected graph data for faster processing...")
    #sampled_data = sample_data(undirected_data, sample_size=10000)

    # Preprocess sampled data
    #adjacency_matrix, nodes = preprocess_undirected_graph(sampled_data)

    # Optionally reduce dimensionality
    reduced_data = reduce_dimensionality(adjacency_matrix, n_components=2)

    # Visualize results for undirected graph
    print("\n~~~~~ Undirected Graph K-Means Clustering (com-dblp.ungraph.txt) ~~~~~")
    kmeans_results(reduced_data, visualize=False)

