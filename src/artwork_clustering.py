"""
Classes and functions for clustering artworks.
"""

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from umap import UMAP

from clip_finetuning import ImageCaptionDataset

from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import warnings

# Setting some things up
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"font.family": "Liberation Sans"})


device = "cuda" if torch.cuda.is_available() else "cpu"



def load_model(base_model: str, model_path: str) -> nn.Module:
    """
    Loads the finetuned model.

    Args:
        base_model (str): The base model to use.
        model_path (str): The path to the finetuned model.

    Returns:
        nn.Module: The finetuned model.
    """
    model, _ = clip.load(base_model, device=device, jit=False)
    if not model_path:
        model.float()
        model.eval()
        return model
    # Loading a finetuned version
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.float()
    model.eval()
    return model


class EmbeddingDatasetBuilder:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = "models/finetuned-v2.pt",
                 raw_dataset: ImageCaptionDataset = ImageCaptionDataset(path_only=True),
                 dataset: ImageCaptionDataset = ImageCaptionDataset(),
                 use_base_model: bool = False) -> None:
        """
        Initializes the EmbeddingDatasetBuilder.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to "models/finetuned.pt".
            raw_dataset (ImageCaptionDataset): The raw dataset. Defaults to ImageCaptionDataset(path_only=True).
            dataset (ImageCaptionDataset): The dataset. Defaults to ImageCaptionDataset().
            use_base_model (bool): Whether to select the base model. Defaults to False.
        """
        self._base_model, _ = clip.load(base_model, device=device, jit=False)
        self._base_model.float()
        self._base_model.eval()
        self._finetuned_model = load_model(base_model, model_path)

        self._raw_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False)
        self._data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        self._use_base_model = use_base_model
    

    def __call__(self) -> pd.DataFrame:
        """
        Builds the embedding dataset.

        Returns:
            pd.DataFrame: The embedding dataset.
        """
        self._base_model.eval()
        self._finetuned_model.eval()
        rows = []

        with torch.no_grad():
            for raw_data, data in zip(self._raw_loader, self._data_loader):
                features = {}
                image_path, _ = raw_data
                image, text = data
                image, text = image.to(device), clip.tokenize(text).to(device)

                features["image_path"] = image_path
                features["text"] = text
                # Adding the image embedding
                image_embedding = self._finetuned_model.encode_image(image)
                if self._use_base_model:
                    image_embedding = self._base_model.encode_image(image)
                image_embedding = image_embedding.cpu().numpy().flatten()
                features["embedding"] = np.array2string(image_embedding, np.inf, separator=",")

                rows.append(features)
        return pd.DataFrame(rows)



class ArtworkClusterer:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = None,
                 dataset_path: str = "data/finetuned_embeddings.csv",
                 signifiers_path: str = "data/signifiers.pkl") -> None:
        """
        Initializes the ArtworkClusterer.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to None.
            dataset_path (str): The path to the embeddings csv. Defaults to "data/finetuned_embeddings.csv".
            signifiers_path (str): The path to the signifiers. Defaults to "data/signifiers.pkl".
        """
        self._model = load_model(base_model, model_path)
        if model_path:
            print(f"Model loaded from [{model_path}].")
        # Loading the embeddings
        dataset = pd.read_csv(dataset_path)
        embeddings = dataset["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        embeddings = np.vstack(embeddings.values)
        # Normalizing
        X_torch = torch.from_numpy(embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        self._embeddings = X_normalized.cpu().numpy()

        self._clusterer = None
        self._labels = None
        self._centroids = []
        self._interps = []
        # Loading signifiers
        with open(signifiers_path, "rb") as f:
            self._signifiers_groups = pickle.load(f) # List[List[str]]
    

    def cluster(self,
                method: str = "kmeans",
                n_terms: int = 5,
                **kwds) -> None:
        """
        Clusters the embeddings using the specified method.

        Args:
            method (str): The clustering method to use. Defaults to "kmeans".
            n_terms (int): The number of terms to use. Defaults to 5.
        
        Returns:
            None
        """
        match method:
            case "kmeans":
                self._clusterer = KMeans(
                    n_clusters=kwds["n_clusters"] if "n_clusters" in kwds else 10,
                    init="k-means++",
                    n_init=10,
                    max_iter=10000,
                    random_state=42
                )
            case "dbscan":
                self._clusterer = DBSCAN(
                    eps=kwds["eps"] if "eps" in kwds else .2,
                    min_samples=kwds["min_samples"] if "min_samples" in kwds else 128,
                    metric="cosine"
                )
        # Fitting the model
        self._labels = self._clusterer.fit_predict(self._embeddings)
        self._get_cluster_centroids()
        self._signify_clusters(n_terms=n_terms)

        # Saving stats and interps
        with open(f"results/{method}.pkl", "wb") as f:
            pickle.dump({
                "stats": self._stats(),
                "interps": self._interps,
                "interps_stats": self._interps_stats()
            }, f)
        self._visualize_embedding_space(method)

    def _stats(self) -> Dict[str, float]:
        """
        Gets the clustering statistics.

        Returns:
            Dict[str, float]: The clustering statistics.
        """
        return {
            "labels": self._labels,
            "sizes": np.bincount(self._labels[self._labels != -1]),
            "silhouette": silhouette_score(self._embeddings, self._labels),
            "calinski_harabasz": calinski_harabasz_score(self._embeddings, self._labels),
            "inertia": self._clusterer.inertia_ if hasattr(self._clusterer, "inertia_") else None
        }
    
    def _interps_stats(self) -> Dict[str, List[float]]:
        """
        Gets the signification statistics, i.e. the average overlap between clusters.

        Returns:
            Dict[str, List[float]]: The signification statistics.
        """
        def avg_overlap(group_idx: int, cluster_idx: int) -> float:
            """
            Gets the average overlap between clusters for a specific group.

            Args:
                group_idx (int): The index of the group.
                cluster_idx (int): The index of the cluster.

            Returns:
                float: The average overlap between clusters.
            """
            group_interp = self._cluster_interps[cluster_idx][group_idx]
            group_terms, n_terms = set([t for t, _ in group_interp]), len(group_interp)
            overlap = 0
            for i, cluster_interp in enumerate(n_clusters):
                if i == cluster_idx:
                    continue
                same_group_terms = set([t for t, _ in cluster_interp[group_idx]])
                overlap += len(same_group_terms.intersection(group_terms)) / n_terms
            return overlap / (n_clusters - 1)
        
        stats = {
            "avg_overlap_per_group": [],
            "avg_overlap_per_cluster": []
        }
        n_clusters, n_groups = len(self._cluster_interps), len(self._signifiers_groups)
        overlaps = np.zeros((n_clusters, n_groups))
        for i in range(n_clusters):
            for j in range(n_groups):
                overlaps[i, j] = avg_overlap(j, i)
        
        stats["avg_overlap_per_group"] = np.mean(overlaps, axis=0).tolist()
        stats["avg_overlap_per_cluster"] = np.mean(overlaps, axis=1).tolist()
        return stats

    def _get_cluster_centroids(self) -> None:
        """
        Gets the centroids for each of the clusters found.

        Returns:
            None
        """
        if isinstance(self._clusterer, KMeans):
            self._centroids = self._clusterer.cluster_centers_
        else:
            # Computing the centroids
            unique_labels = np.unique(self._labels[self._labels != -1])
            self._centroids = [self._embeddings[self._labels == label].mean(axis=0) for label in unique_labels]
    
    def _signify_clusters(self, n_terms: int = 5) -> None:
        """
        Signifies the clusters found using the cluster centroids.

        Args:
            n_terms (int): The number of terms to use. Defaults to 5.

        Returns:
            None
        """
        with torch.no_grad():
            for centroid in self._centroids:
                interp = []
                centroid = torch.from_numpy(centroid).float().to(device)
                # Iterating over the groups
                for group in self._signifiers_groups:
                    signifiers = torch.cat([clip.tokenize(s) for _, s in group]).to(device)
                    signifiers = self._model.encode_text(signifiers)
                    signifiers = signifiers / signifiers.norm(dim=-1, keepdim=True)
                    # Computing the cosine similarity
                    similarity = 100.0 * centroid @ signifiers.t()

                    values, indices = similarity.topk(min(n_terms, len(group)))
                    group_interp = [(group[i][0], v.item()) for i, v in zip(indices, values)]
                    interp.append(group_interp)

                self._interps.append(interp)
    
    def _visualize_embedding_space(self, method: str) -> None:
        """
        Visualizes the embedding space.

        Args:
            method (str): The clustering method to use.

        Returns:
            None
        """
        reducer = UMAP(
            n_neighbors=10,
            min_dist=.01,
            spread=1.5,
            metric="cosine",
            random_state=42
        )
        embeddings = reducer.fit_transform(self._embeddings)
        sample = train_test_split(embeddings, self._labels, train_size=.01, stratify=self._labels, random_state=42)
        sampled_embeddings, sampled_labels = sample[0], sample[2]
        # Plotting
        plt.figure(figsize=(10, 10))
        plt.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1], c=sampled_labels, cmap="viridis", s=5, alpha=.8)
        plt.title(f"Embedding Space clustered with {method.upper()}")
        plt.savefig(f"results/{method}.svg", format="svg")
        plt.close()
