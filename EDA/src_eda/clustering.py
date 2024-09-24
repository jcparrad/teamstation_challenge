# EDA/src_eda/clustering.py

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import hdbscan
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, df: pd.DataFrame, subset = ['Question_embedding', 'Answer_Solution_embedding']):
        self.df = df.dropna(subset=subset)

    def extract_question_embeddings(self):
        return np.array(self.df['Question_embedding'].tolist())

    def extract_answer_embeddings(self):
        return np.array(self.df['Answer_Solution_embedding'].tolist())
    
    #def get_concat_embeddings(self, subset = ['Question_embedding', ]):
    
    

class ClusteringAnalysis:
    def __init__(self, embeddings, min_cluster_size, df, cluster_label):
        self.embeddings = embeddings
        self.min_cluster_size = min_cluster_size
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, 
                                         gen_min_span_tree=True,
                                         min_samples = 2,)
        self.labels = None
        self.df = df
        self.cluster_label = cluster_label  # Store the label for the cluster

    def perform_clustering(self):
        self.labels = self.clusterer.fit_predict(self.embeddings)
        return self.labels

    def label_clusters(self):
        # Add cluster labels to the DataFrame with custom names
        self.df[self.cluster_label] = self.labels

class TSNEVisualizer:
    def __init__(self, embeddings, labels, title):
        self.embeddings = embeddings
        self.labels = labels
        self.title = title

    def plot_clusters(self):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(self.embeddings)

        plt.figure(figsize=(10, 7))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=self.labels, cmap='viridis', alpha=0.5)
        plt.title(self.title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Cluster Label')
        plt.show()

    def plot_minimum_spanning_tree(self, clusterer):
        plt.figure(figsize=(10, 7))
        clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6,
                                               node_size=20, edge_linewidth=2)
        plt.title("Minimum Spanning Tree")
        plt.show()

    def plot_single_linkage_tree(self, clusterer):
        plt.figure(figsize=(10, 7))
        clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        plt.title("Single Linkage Tree")
        plt.show()

    def plot_condensed_tree(self, clusterer, select_clusters=False):
        plt.figure(figsize=(10, 7))
        clusterer.condensed_tree_.plot(select_clusters=select_clusters,
                                        selection_palette=sns.color_palette())
        plt.title("Condensed Tree" + (" (Selected Clusters)" if select_clusters else ""))
        plt.show()

