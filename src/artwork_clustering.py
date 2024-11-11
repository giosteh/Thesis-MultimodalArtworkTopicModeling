"""
Classes and functions for clustering artworks.
"""

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        self._reducer = None

        with open(signifiers_path, "rb") as f:
            self._groups_of_signifiers = pickle.load(f) # List[List[str]]
    

    def cluster(self,
                method: str = "kmeans",
                reduce_with: str = None,
                represent_with: str = "centroid",
                n_terms: int = 5,
                **kwargs) -> None:
        """
        Clusters the embeddings using the specified mode.

        Args:
            method (str): The clustering method to use. Defaults to "kmeans".
            reduce_with (str): The method to use for dimensionality reduction. Defaults to None.
            represent_with (str): The method to use to represent the clusters. Defaults to "centroid".
            n_terms (int): The number of terms to use. Defaults to 5.

        Returns:
            None
        """
        clusterer = None
        match method:
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
                    min_samples=kwargs["min_samples"] if "min_samples" in kwargs else 20,
                    metric="cosine"
                )
            case "birch":
                clusterer = Birch(
                    n_clusters=kwargs["n_clusters"] if "n_clusters" in kwargs else 10,
                    threshold=kwargs["threshold"] if "threshold" in kwargs else 1.0,
                    branching_factor=kwargs["branching_factor"] if "branching_factor" in kwargs else 150
                )
        
        reducer = None
        match reduce_with:
            case "umap":
                reducer = UMAP(
                    n_components=kwargs["n_components"] if "n_components" in kwargs else 2,
                    n_neighbors=kwargs["n_neighbors"] if "n_neighbors" in kwargs else 30,
                    min_dist=kwargs["min_dist"] if "min_dist" in kwargs else .1,
                    metric="cosine",
                    random_state=42,
                    n_jobs=1
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
            self._reducer = reducer
        
        inertia = None
        self._labels = clusterer.fit_predict(self._embeddings)
        if method == "kmeans":
            inertia = clusterer.inertia_
        # Getting cluster representatives and interpretations        
        cluster_reprs = self._get_cluster_reprs(method, represent_with)
        interpretations = self._signify_clusters(cluster_reprs, n_terms)

        stats = self._get_stats()
        stats["inertia"] = inertia
        # Saving results
        with open("res/interp.pkl", "wb") as f:
            data = {
                "stats": stats,
                "interpretations": interpretations
            }
            pickle.dump(data, f)
        self._visualize(self._labels)
    
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
            "calinski_harabasz": calinski_harabasz_score(self._embeddings, self._labels)
        }
    
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
                        if self._reducer:
                            signifiers = self._reducer.transform(signifiers.cpu().numpy())
                            signifiers = torch.from_numpy(signifiers).float().to(device)
                        # Compute similarity
                        similarity = (100.0 * cluster_repr @ signifiers.t()).softmax(dim=-1)

                        values, indices = similarity.topk(min(n_terms, len(group)))
                        interpretation = [(group[i][0], v.item()) for i, v in zip(indices, values)]
                        interpretations.append(interpretation)

                cluster_interps.append(interpretations)
        return cluster_interps
    
    def _visualize(self, labels: np.ndarray, n_neighbors: int = 50, min_dist: float = .1) -> None:
        """
        Visualizes the clusters using UMAP.

        Args:
            labels (np.ndarray): The labels.
            n_neighbors (int): The number of neighbors to use for UMAP. Defaults to 50.
            min_dist (float): The minimum distance to use for UMAP. Defaults to .1.

        Returns:
            None
        """
        umap_reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            metric="cosine",
            n_jobs=1
        )
        pca_reducer = PCA(
            n_components=2,
            random_state=42
        )

        embeddings2d_umap = umap_reducer.fit_transform(self._embeddings)
        plt.figure(figsize=(10, 7))
        plt.scatter(embeddings2d_umap[:, 0], embeddings2d_umap[:, 1], c=labels, cmap="viridis", s=1.5)
        plt.colorbar(label="Cluster labels")
        plt.title("UMAP visualization")
        plt.savefig("res/visual_umap.png", dpi=300, bbox_inches="tight")

        embeddings2d_pca = pca_reducer.fit_transform(self._embeddings)
        plt.figure(figsize=(10, 7))
        plt.scatter(embeddings2d_pca[:, 0], embeddings2d_pca[:, 1], c=labels, cmap="viridis", s=1.5)
        plt.colorbar(label="Cluster labels")
        plt.title("PCA visualization")
        plt.savefig("res/visual_pca.png", dpi=300, bbox_inches="tight")

        # Close the plot
        plt.close()