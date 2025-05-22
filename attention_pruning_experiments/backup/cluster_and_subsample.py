from src.utils import set_seed
import os 
import json 
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import argparse 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.dpp import select_k_samples_with_dpp_from_dataset
from collections import Counter
import numpy as np
import string
def check_and_download_math_500(dataset_path='datasets/math500/test.jsonl', split='test'):
    # download the dataset if it doesn't exist
    url = f"https://media.githubusercontent.com/media/openai/prm800k/refs/heads/main/prm800k/math_splits/{split}.jsonl"
    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        os.system(f"wget {url} -O {dataset_path}")
    else:
        print(f"Loaded dataset from {dataset_path}.")
        
# TODO: merge into src/math_500_utils.py
def load_dataset(dataset_path, split='test'):
    dataset_path = os.path.join(dataset_path, f'{split}.jsonl')
    check_and_download_math_500(dataset_path, split=split)
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_jsonl(dataset_path):
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def embed_data(text_list, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_list, convert_to_numpy=True)
    return embeddings


class KmeansCluster:
    def __init__(self, model_name='all-MiniLM-L6-v2', n_clusters=5):
        """
        Base class for clustering using SentenceTransformer and KMeans.
        
        Parameters:
        - model_name: Name of the SentenceTransformer model for embeddings
        - n_clusters: Number of clusters for KMeans
        """
        self.model = SentenceTransformer(model_name)
        self.clusterer = None
        self.n_clusters = n_clusters
        self.clusters = None

    def fit(self, text_list, embeddings=None):
        """
        Fit the KMeans clustering model.
        
        Parameters:
        - text_list: List of text inputs
        - embeddings: (Optional) Precomputed embeddings for the text inputs
        """
        print("Fitting the clustering model...")
        if embeddings is None:
            embeddings = self.model.encode(text_list, convert_to_numpy=True)
        
        # Initialize and fit the KMeans clusterer
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters = self.clusterer.fit_predict(embeddings)

    def save(self, clusterer_path="kmeans_clusterer.pkl", clusters_path="clusters.npy"):
        """
        Save the clusterer and cluster labels to files.
        
        Parameters:
        - clusterer_path: Path to save the KMeans model
        - clusters_path: Path to save the cluster labels
        """
        os.makedirs(os.path.dirname(clusterer_path), exist_ok=True)
        os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
        with open(clusterer_path, "wb") as f:
            pickle.dump(self.clusterer, f)
        np.save(clusters_path, self.clusters)

    def load(self, clusterer_path="kmeans_clusterer.pkl", clusters_path="clusters.npy"):
        """
        Load the clusterer and cluster labels from files.
        
        Parameters:
        - clusterer_path: Path to load the KMeans model
        - clusters_path: Path to load the cluster labels
        """
        print(f"Loading the clustering model from {clusterer_path} and {clusters_path}...")
        with open(clusterer_path, "rb") as f:
            self.clusterer = pickle.load(f)
        self.clusters = np.load(clusters_path)

    def predict(self, text_list, embeddings=None):
        """
        Predict cluster labels for new data.
        
        Parameters:
        - text_list: List of text inputs
        - embeddings: (Optional) Precomputed embeddings for the text inputs
        
        Returns:
        - Cluster labels for the input data
        """
        if embeddings is None:
            embeddings = self.model.encode(text_list, convert_to_numpy=True)
        return self.clusterer.predict(embeddings)

    def get_cluster_stats(self, all_subjects=None):
        """
        Get statistics about the clusters.
        
        Returns:
        - n_clusters: Number of clusters
        - cluster_counts: Dictionary of cluster label counts
        """
        unique, counts = np.unique(self.clusters, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        
        subject_counts_by_cluster = {}
        if all_subjects is not None:
            for cluster_label in unique:
                cluster_indices = np.where(self.clusters == cluster_label)[0]
                subjects_in_cluster = [all_subjects[i] for i in cluster_indices]
                subject_counts_by_cluster[cluster_label] = dict(zip(*np.unique(subjects_in_cluster, return_counts=True)))
        
        return len(unique), cluster_counts, subject_counts_by_cluster

    def visualize(self, image_path, embeddings, title="Cluster Visualization"):
        """
        Visualize the clusters using UMAP.
        
        Parameters:
        - image_path: Path to save the visualization
        - embeddings: Embeddings of the data
        - title: Title for the visualization
        """
        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(embeddings)

        # Step 2: Plot the clusters
        plt.figure(figsize=(12, 8))
        unique_clusters = set(self.clusters)

        # Assign colors to clusters
        palette = sns.color_palette('hsv', len(unique_clusters))
        colors = [palette[cluster] for cluster in self.clusters]

        # Scatter plot for the embeddings
        plt.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1], 
            c=colors, 
            s=10, 
            alpha=0.8
        )

        # Add legend
        for cluster_label in unique_clusters:
            label = f"Cluster {cluster_label}"
            plt.scatter([], [], c=palette[cluster_label], label=label)

        plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title, fontsize=16)
        plt.xlabel("PCA Dimension 1", fontsize=12)
        plt.ylabel("PCA Dimension 2", fontsize=12)
        plt.tight_layout()
        plt.savefig(image_path, dpi=200)

    def visualize_subject_and_cluster_distribution(self, name1, name2, subject_counts_by_cluster):
        """
        Visualize two distributions:
        1. Subject distribution by cluster (stacked bar chart).
        2. Cluster distribution by subject (stacked bar chart).

        Parameters:
        - name1: Filename to save the first plot (subject distribution by cluster).
        - name2: Filename to save the second plot (cluster distribution by subject).
        - subject_counts_by_cluster: Dictionary where keys are cluster labels, and values
        are dictionaries of subject counts for that cluster.

        Saves:
        - Two stacked bar charts as images.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Prepare data for visualization
        clusters = list(subject_counts_by_cluster.keys())
        subjects = set()  # Unique subjects across all clusters
        for cluster_subjects in subject_counts_by_cluster.values():
            subjects.update(cluster_subjects.keys())
        subjects = sorted(subjects)  # Sort subjects for consistent ordering

        # Create a matrix where rows are clusters and columns are subject counts
        counts_matrix = []
        for cluster in clusters:
            counts = [subject_counts_by_cluster[cluster].get(subject, 0) for subject in subjects]
            counts_matrix.append(counts)

        counts_matrix = np.array(counts_matrix)  # Convert to numpy array

        # === Plot 1: Subject distribution by cluster ===
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.6
        x = np.arange(len(clusters))  # Cluster indices

        # Plot stacked bars
        bottom = np.zeros(len(clusters))
        for i, subject in enumerate(subjects):
            ax.bar(
                x,
                counts_matrix[:, i],
                label=subject,
                bottom=bottom,
                width=bar_width
            )
            bottom += counts_matrix[:, i]

        # Add labels and legend
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cluster {c}" if c != -1 else "Noise" for c in clusters])
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_xlabel("Clusters", fontsize=12)
        ax.set_title("Subject Distribution by Cluster", fontsize=16)
        ax.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save the first plot
        plt.tight_layout()
        plt.savefig(name1, dpi=200)
        plt.close()

        # === Plot 2: Cluster distribution by subject (as percentages) ===
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.6
        x = np.arange(len(subjects))  # Subject indices

        # Transpose the data: rows become subjects, columns become clusters
        counts_matrix_transposed = counts_matrix.T

        # Normalize data to percentages
        percentages_matrix = counts_matrix_transposed / counts_matrix_transposed.sum(axis=1, keepdims=True) * 100

        # Plot stacked bars
        bottom = np.zeros(len(subjects))
        for i, cluster in enumerate(clusters):
            ax.bar(
                x,
                percentages_matrix[:, i],
                label=f"Cluster {cluster}" if cluster != -1 else "Noise",
                bottom=bottom,
                width=bar_width
            )
            bottom += percentages_matrix[:, i]

        # Add labels and legend
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=45, ha="right")
        ax.set_ylabel("Percentage of Samples (%)", fontsize=12)
        ax.set_xlabel("Subjects", fontsize=12)
        ax.set_title("Cluster Distribution by Subject (Percentage)", fontsize=16)
        ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save the second plot
        plt.tight_layout()
        plt.savefig(name2, dpi=200)
        plt.close()

def save_subsampled_data_to_file(subsampled_data, save_path):
    """
    Save subsampled data to a JSONL file.
    """
    with open(save_path, "w") as f:
        for item in subsampled_data:
            f.write(json.dumps(item) + "\n")
    print(f"Subsampled data saved to {save_path}")

def load_subsampled_data(file_path):
    """
    Load subsampled data from a JSONL file.
    """
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_repo", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--sbert_model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--n_clusters", type=int, default=7) 
    parser.add_argument("--subsample_size", type=int, default=10)
    parser.add_argument("--recluster", type=bool, default=True)
    args = parser.parse_args()
    
    set_seed(args.seed)
    cluster_path = f"shapley_prune/tmp/kmeans_clusterer_{args.n_clusters}.pkl"
    clusters_path = f"shapley_prune/tmp/k_means_clusters_{args.n_clusters}.npy"
    data = load_jsonl('datasets/math500/train_updated.jsonl')
    train_problems = [x['problem'] for x in data]
    train_subjects = [x['subject'] for x in data]
    train_answers = [x['answer'] for x in data]
    question_embeddings = embed_data(train_problems, model_name=args.sbert_model_name)
    
    clustering_model = KmeansCluster(model_name=args.sbert_model_name, n_clusters=args.n_clusters)
    if args.recluster or not os.path.exists(cluster_path) or not os.path.exists(clusters_path):
        clustering_model.fit(train_problems, embeddings=question_embeddings)
        clustering_model.save(clusterer_path=cluster_path, clusters_path=clusters_path)
    else:
        clustering_model.load(clusterer_path=cluster_path, clusters_path=clusters_path)
    
    clustering_model.visualize("cluster_visualization.png", question_embeddings, title="Cluster Visualization")
    
    new_problem = "Evaluate log(100)."
    predicted_cluster = clustering_model.predict([new_problem])
    print(f"Predicted cluster for the new problem: {predicted_cluster}")
    n_clusters, cluster_count, subject_count_by_cluster = clustering_model.get_cluster_stats(all_subjects=train_subjects)
    print(f"Number of clusters: {n_clusters}")
    print(f"cluster_count: {cluster_count}")
    for k, v in subject_count_by_cluster.items():
        print(f"Cluster {k} subject count: {v}")
    
        
    clustering_model.visualize_subject_and_cluster_distribution(f"shapley_prune/tmp/cluster_{args.n_clusters}_subject_distribution.png", 
                                              f"shapley_prune/tmp/cluster_{args.n_clusters}_cluster_distribution.png", 
                                              subject_count_by_cluster)
    clusters = clustering_model.clusters
    subsampled_data = []
    for cluster_id in range(args.n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_embeddings = question_embeddings[cluster_indices]
        
        if len(cluster_indices) > args.subsample_size:
            selected_indices = select_k_samples_with_dpp_from_dataset(cluster_embeddings, args.subsample_size)
            subsampled_indices = cluster_indices[selected_indices]
        else:
            subsampled_indices = cluster_indices  # Use all if fewer than subsample_size
        
        subsampled_data.extend([{"problem": train_problems[i], "subject": train_subjects[i], "answer": train_answers[i]} for i in subsampled_indices])
    
    # Save subsampled data to file
    
    # Load subsampled data (test function)
    subsampled_file_path = f"shapley_prune/tmp/subsample{args.subsample_size}_cluster{args.n_clusters}.jsonl"
    save_subsampled_data_to_file(subsampled_data, subsampled_file_path)
    loaded_data = load_subsampled_data(subsampled_file_path)
    print(f"Loaded {len(loaded_data)} subsampled data points.")

    
    # clustering_model.visualize_frequent_words(f"shapley_prune/tmp/cluster_{args.n_clusters}_words_distribution.png", frequent_words_by_cluster)
    # some sample belonging to predicted_cluster
    # sample_indices = np.where(clustering_model.clusters == predicted_cluster)[0]
    # print("Sample problems from the predicted cluster:")
    # for idx in sample_indices[:20]:
    #     print(f"Sample {idx}: {train_problems[idx]}\n")
        
    