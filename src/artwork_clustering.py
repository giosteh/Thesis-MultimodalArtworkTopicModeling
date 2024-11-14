"""
Classes and functions for clustering artworks.
"""

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from umap import UMAP

from clip_finetuning import ImageCaptionDataset

from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import warnings

# Ignoring warnings relative to model loading (FutureWarnings)
warnings.filterwarnings("ignore", category=FutureWarning)


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
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.float()
    model.eval()

    return model


class EmbeddingDatasetBuilder:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = "models/finetuned-v2.pt",
                 raw_dataset: ImageCaptionDataset = ImageCaptionDataset(raw_only=True),
                 dataset: ImageCaptionDataset = ImageCaptionDataset(),
                 use_base_model: bool = False) -> None:
        """
        Initializes the EmbeddingDatasetBuilder.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to "models/finetuned.pt".
            raw_dataset (ImageCaptionDataset): The raw dataset. Defaults to ImageCaptionDataset(raw_only=True).
            dataset (ImageCaptionDataset): The dataset. Defaults to ImageCaptionDataset().
            use_base_model (bool): Whether to select the base embeddings. Defaults to False.
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

                base_features = self._base_model.encode_image(image)
                base_features = base_features.cpu().numpy().flatten()
                finetuned_features = self._finetuned_model.encode_image(image)
                finetuned_features = finetuned_features.cpu().numpy().flatten()

                features["image_path"] = image_path
                features["text"] = text
                features["base_embedding"] = np.array2string(base_features, np.inf, separator=",")
                features["finetuned_embedding"] = np.array2string(finetuned_features, np.inf, separator=",")

                rows.append(features)
        
        if self._use_base_model:
            return pd.DataFrame(rows).rename(columns={"base_embedding": "embedding"})
        return pd.DataFrame(rows).rename(columns={"finetuned_embedding": "embedding"})


class ArtworkClusterer:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = None,
                 dataset: pd.DataFrame = EmbeddingDatasetBuilder()(),
                 signifiers_path: str = "data/signifiers.pkl") -> None:
        """
        Initializes the ArtworkClusterer.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to None.
            dataset (pd.DataFrame): The dataset. Defaults to EmbeddingDatasetBuilder()().
            signifiers_path (str): The path to the signifiers. Defaults to "data/signifiers.pkl".
        
        Returns:
            None
        """
        self._model, _ = clip.load(base_model, device=device, jit=False)
        self._model.float()
        self._model.eval()
        if model_path:
            self._model = load_model(base_model, model_path)
        
        embeddings = dataset["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        embeddings = np.vstack(embeddings.values)
        # Normalizing
        X_torch = torch.from_numpy(embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)

        self._embeddings = X_normalized.cpu().numpy()
        self._labels = None
        self._clusterer = None
        with open(signifiers_path, "rb") as f:
            self._groups_of_signifiers = pickle.load(f) # List[List[str]]
    

    def cluster(self,
                method: str = "kmeans",
                represent_with: str = "centroid",
                n_terms: int = 5,
                **kwargs) -> None:
        """
        Clusters the embeddings using the specified method.

        Args:
            method (str): The clustering method to use. Defaults to "kmeans".
            represent_with (str): The method to use to represent the clusters. Defaults to "centroid".
            n_terms (int): The number of terms to use. Defaults to 5.
        
        Returns:
            None
        """
        points = self._embeddings
        match method:
            case "kmeans":
                self._clusterer = KMeans(
                    n_clusters=kwargs["n_clusters"] if "n_clusters" in kwargs else 10,
                    init="k-means++",
                    n_init=10,
                    max_iter=1000,
                    random_state=42
                )
            case "dbscan":
                self._clusterer = DBSCAN(
                    eps=kwargs["eps"] if "eps" in kwargs else 0.2,
                    min_samples=kwargs["min_samples"] if "min_samples" in kwargs else 20,
                    metric="cosine"
                )
            case "birch":
                self._clusterer = Birch(
                    n_clusters=kwargs["n_clusters"] if "n_clusters" in kwargs else 10,
                    threshold=kwargs["threshold"] if "threshold" in kwargs else 1.0,
                    branching_factor=kwargs["branching_factor"] if "branching_factor" in kwargs else 200
                )
            case "birch+kmeans":
                birch = Birch(
                    n_clusters=None,
                    threshold=kwargs["threshold"] if "threshold" in kwargs else 0.5,
                    branching_factor=kwargs["branching_factor"] if "branching_factor" in kwargs else 50
                )
                birch.fit(points)
                points = birch.subcluster_centers_
                self._clusterer = KMeans(
                    n_clusters=kwargs["n_clusters"] if "n_clusters" in kwargs else 10,
                    init="k-means++",
                    n_init=10,
                    max_iter=1000,
                    random_state=42
                )
        # Clustering & signification
        self._labels = self._clusterer.fit_predict(points)
        cluster_reprs = self._get_cluster_reprs(represent_with)
        cluster_interps = self._signify_clusters(cluster_reprs, n_terms)
        # Visualizing
        self._visualize_with_umap(method)
        # Saving results
        with open(f"results/{method}.pkl", "wb") as f:
            data = {
                "interps": cluster_interps,
                "stats": self._get_stats()
            }
            pickle.dump(data, f)
    
    def _get_stats(self) -> Dict[str, float]:
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
    
    def _get_cluster_reprs(self, mode: str) -> List[torch.Tensor]:
        """
        Gets the cluster representatives.

        Args:
            clustering_method (str): The clustering method used ("centroid" or "medoid").
            mode (str): The mode to use to get the cluster representatives.

        Returns:
            List[torch.Tensor]: The cluster representatives.
        """
        unique_labels = np.unique(self._labels[self._labels != -1])
        cluster_reprs = []

        for label in unique_labels:
            mask = (self._labels == label)
            points = self._embeddings[mask]

            centroid = np.mean(points, axis=0)
            if mode == "centroid":
                cluster_reprs.append(torch.from_numpy(centroid).float())
            else:
                distances = cdist(points, points)
                total_distances = np.sum(distances, axis=1)
                medoid = points[total_distances.argmin()]

                cluster_reprs.append(torch.from_numpy(medoid).float())
        return cluster_reprs
    
    def _signify_clusters(self, cluster_reprs: List[torch.Tensor], n_terms: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Signifies the clusters found by the model.

        Args:
            cluster_reprs (List[torch.Tensor]): The cluster representatives.
            n_terms (int): The number of terms to use. Defaults to 5.

        Returns:
            List[List[Tuple[str, float]]]: The interpretations.
        """
        cluster_interps = []

        with torch.no_grad():
            for cluster_repr in cluster_reprs:
                interpretations = []
                cluster_repr = cluster_repr.to(device)

                for group in self._groups_of_signifiers:
                    if len(group) > 0:
                        signifiers = torch.cat([clip.tokenize(s) for _, s in group]).to(device)
                        signifiers = self._model.encode_text(signifiers)
                        signifiers = signifiers / signifiers.norm(dim=-1, keepdim=True)
                        # Compute similarity
                        similarity = 100.0 * cluster_repr @ signifiers.t()

                        values, indices = similarity.topk(min(n_terms, len(group)))
                        interpretation = [(group[i][0], v.item()) for i, v in zip(indices, values)]
                        interpretations.append(interpretation)

                cluster_interps.append(interpretations)
        return cluster_interps
    
    def _visualize_with_umap(self, method: str, n_neighbors: int = 15, min_dist: float = 0.1) -> None:
        """
        Visualizes the clusters found by the model using UMAP.

        Args:
            method (str): The clustering method used.
            n_neighbors (int): The number of neighbors to use. Defaults to 15.
            min_dist (float): The minimum distance to use. Defaults to 0.1.
        
        Returns:
            None
        """
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
            n_jobs=1
        )
        embeddings_reduced = reducer.fit_transform(self._embeddings)
        # Visualizing
        plt.figure(figsize=(10, 7))
        plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], c=self._labels, cmap="viridis", s=1.5, alpha=.7)
        plt.title(f"Clusters found by {method} visualized with UMAP")
        plt.colorbar(label="Cluster label")
        plt.savefig(f"results/{method}.svg", format="svg", bbox_inches="tight")
        plt.close()
