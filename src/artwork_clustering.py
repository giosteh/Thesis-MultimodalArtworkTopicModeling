"""
Classes and functions for clustering artworks.
"""

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from hdbscan import HDBSCAN
from umap import UMAP

from clip_finetuning import ImageCaptionDataset

from typing import List, Tuple
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


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
        X_torch = torch.from_numpy(embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        self._embeddings = X_normalized.cpu().numpy()
        self._labels = None
        self._probabilities = None

        with open(signifiers_path, "rb") as f:
            self._signifiers = pickle.load(f)
    

    def cluster(self,
                method: str = "kmeans",
                reduce_with: str = None,
                represent_with: str = "centroid",
                n_terms: int = 10,
                **kwargs) -> None:
        """
        Clusters the embeddings using the specified mode.

        Args:
            method (str): The clustering method to use. Defaults to "kmeans".
            reduce_with (str): The method to use for dimensionality reduction. Defaults to None.
            represent_with (str): The method to use to represent the clusters. Defaults to "centroid".
            n_terms (int): The number of terms to use. Defaults to 10.

        Returns:
            None
        """
        clusterer = None
        match method: # Clustering method
            case "kmeans":
                clusterer = KMeans(
                    n_clusters=kwargs["n_clusters"] if "n_clusters" in kwargs else 10,
                    init="k-means++",
                    n_init=10,
                    max_iter=1000,
                    random_state=42
                )
            case "dbscan":
                clusterer = DBSCAN(
                    eps=kwargs["eps"] if "eps" in kwargs else 0.2,
                    min_samples=kwargs["min_samples"] if "min_samples" in kwargs else 5,
                    metric="cosine"
                )
            case "hdbscan":
                clusterer = HDBSCAN(
                    min_cluster_size=kwargs["min_cluster_size"] if "min_cluster_size" in kwargs else 10,
                    min_samples=kwargs["min_samples"] if "min_samples" in kwargs else 5,
                    metric="cosine",
                    cluster_selection_method="eom",
                    prediction_data=True
                )
        
        reducer = None
        match reduce_with: # Dimensionality reduction
            case "umap":
                reducer = UMAP(
                    n_components=kwargs["n_components"] if "n_components" in kwargs else 2,
                    n_neighbors=kwargs["n_neighbors"] if "n_neighbors" in kwargs else 15,
                    min_dist=kwargs["min_dist"] if "min_dist" in kwargs else .1,
                    random_state=42
                )
            case "tsne":
                reducer = TSNE(
                    n_components=kwargs["n_components"] if "n_components" in kwargs else 2,
                    perplexity=kwargs["perplexity"] if "perplexity" in kwargs else 30,
                    random_state=42
                )
            case "pca":
                reducer = PCA(
                    n_components=kwargs["n_components"] if "n_components" in kwargs else 2,
                    random_state=42
                )
        if reducer:
            self._embeddings = reducer.fit_transform(self._embeddings)
        if clusterer:
            self._labels = clusterer.fit_predict(self._embeddings)
            if method == "hdbscan":
                self._probabilities = clusterer.probabilities_
        
        cluster_reprs = self._get_cluster_reprs(method, represent_with)
        interpretations = self._signify_clusters(cluster_reprs, n_terms)
        # Saving and visualizing
        with open("data/results.pkl", "wb") as f:
            data = (self._labels, interpretations)
            pickle.dump(data, f)
        self._visualize(self._labels)
    
    def _get_cluster_reprs(self, clustering_method: str, mode: str) -> List[torch.Tensor]:
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
            weights = self._probabilities[mask] if clustering_method == "hdbscan" else None

            centroid = np.mean(points, axis=0)
            if weights:
                centroid = np.average(points, axis=0, weights=weights)
            
            if mode == "centroid":
                cluster_reprs.append(torch.from_numpy(centroid).float())
            else:
                distances = cdist(points, points)
                if weights:
                    distances = distances * weights.reshape(-1, 1)
                total_distances = np.sum(distances, axis=1)
                medoid = points[total_distances.argmin()]

                cluster_reprs.append(torch.from_numpy(medoid).float())
        return cluster_reprs
    
    def _signify_clusters(self, cluster_reprs: List[torch.Tensor], n_terms: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Signifies the clusters assigning the most similar labels to each centroid.

        Args:
            cluster_reprs (List[torch.Tensor]): The cluster representatives.
            n_labels (int): The number of labels to assign. Defaults to 10.

        Returns:
            List[List[Tuple[str, float]]]: The list of interpretations.
        """
        interpretations = []
        signifiers = torch.cat([clip.tokenize(f"a {s} painting") for s in self._signifiers]).to(device)

        with torch.no_grad():
            for cluster_repr in cluster_reprs:
                cluster_repr = cluster_repr.to(device)
                signifiers = self._model.encode_text(signifiers)
                signifiers = signifiers / signifiers.norm(dim=-1, keepdim=True)

                similarity = (100 * cluster_repr @ signifiers.t()).softmax(dim=-1)
                values, indices = similarity[0].topk(n_terms)
                interpretation = [(self._signifiers[i], v.item()) for i, v in zip(indices, values)]

                interpretations.append(interpretation)
        return interpretations

    def _visualize(self, labels: np.ndarray, n_neighbors: int = 15, min_dist: float = .1) -> None:
        """
        Visualizes the clusters using UMAP.

        Args:
            labels (np.ndarray): The labels.
            n_neighbors (int): The number of neighbors to use for UMAP. Defaults to 30.
            min_dist (float): The minimum distance to use for UMAP. Defaults to .1.

        Returns:
            None
        """
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        embeddings_2d = reducer.fit_transform(self._embeddings)

        plt.figure(figsize=(10, 7))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="Spectral", s=5)
        plt.colorbar(label="Cluster label")
        plt.title("UMAP projection of the clustered embeddings")
        plt.savefig("visual.png", dpi=300, bbox_inches="tight")
        plt.close()
