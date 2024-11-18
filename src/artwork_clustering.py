"""
Classes and functions for clustering artworks.
"""

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from umap import UMAP

from clip_finetuning import ImageCaptionDataset

from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D visualization
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

                features["image_path"] = image_path
                features["text"] = text
                # Adding image embedding
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
        self._model, _ = clip.load(base_model, device=device, jit=False)
        self._model.float()
        self._model.eval()
        if model_path:
            self._model = load_model(base_model, model_path)
            print(f"Model loaded from [{model_path}].")
        # Loading embeddings
        dataset = pd.read_csv(dataset_path)
        embeddings = dataset["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        embeddings = np.vstack(embeddings.values)
        # Normalizing
        X_torch = torch.from_numpy(embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        self._embeddings = X_normalized.cpu().numpy()

        self._labels = None
        self._clusterer = None
        self._cluster_interps = None
        with open(signifiers_path, "rb") as f:
            self._signifiers_groups = pickle.load(f) # List[List[str]]
    

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
                    max_iter=10000,
                    random_state=42
                )
            case "kmedoids":
                self._clusterer = KMedoids(
                    n_clusters=kwargs["n_clusters"] if "n_clusters" in kwargs else 10,
                    init="k-medoids++",
                    metric="cosine",
                    method="pam",
                    max_iter=10000,
                    random_state=42
                )
            case "dbscan":
                self._clusterer = DBSCAN(
                    eps=kwargs["eps"] if "eps" in kwargs else 0.2,
                    min_samples=kwargs["min_samples"] if "min_samples" in kwargs else 128,
                    metric="cosine"
                )
        # Clustering & signification
        self._labels = self._clusterer.fit_predict(points)
        cluster_reprs = self._get_cluster_reprs(represent_with)
        self._cluster_interps = self._signify_clusters(cluster_reprs, n_terms)
        # Visualizing
        self._visualize_with_umap(method)
        # Saving results
        with open(f"results/{method}.pkl", "wb") as f:
            data = {
                "interps": self._cluster_interps,
                "interps_stats": self._interps_stats(),
                "stats": self._stats()
            }
            pickle.dump(data, f)
    
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
        def avg_overlap_per_group(group_idx: int, cluster_idx: int) -> float:
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
            for i, cluster_interp in enumerate(self._cluster_interps):
                if i == cluster_idx:
                    continue
                same_group_terms = set([t for t, _ in cluster_interp[group_idx]])
                overlap += len(same_group_terms.intersection(group_terms)) / n_terms
            return overlap / (len(self._cluster_interps) - 1)
        
        stats = {
            "avg_overlap_per_group": [],
            "avg_overlap_per_cluster": [],
            "overall_avg_per_group": []
        }
        for i in range(len(self._cluster_interps)):
            stats["avg_overlap_per_group"].append([])
            for j in range(len(self._signifiers_groups)):
                stats["avg_overlap_per_group"][-1].append(avg_overlap_per_group(j, i))
            stats["avg_overlap_per_cluster"].append(np.mean(stats["avg_overlap_per_group"][-1]))
        
        for j in range(len(self._signifiers_groups)):
            stats["overall_avg_per_group"].append(np.mean([stats["avg_overlap_per_group"][i][j] for i in range(len(self._cluster_interps))]))
        return stats
    
    def _get_cluster_reprs(self, mode: str) -> List[torch.Tensor]:
        """
        Gets the cluster representatives.

        Args:
            mode (str): The mode to use to get the cluster representatives.

        Returns:
            List[torch.Tensor]: The cluster representatives.
        """
        unique_labels = np.unique(self._labels[self._labels != -1])
        cluster_reprs = []

        for label in unique_labels:
            mask = (self._labels == label)
            points = self._embeddings[mask]

            if mode == "centroid":
                centroid = np.mean(points, axis=0)
                cluster_reprs.append(torch.from_numpy(centroid).float())
            else:
                distances = cdist(points, points, metric="cosine")
                total_distances = np.sum(distances, axis=1)
                medoid = points[total_distances.argmin()]
                cluster_reprs.append(torch.from_numpy(medoid).float())
        return cluster_reprs
    
    def _signify_clusters(self, cluster_reprs: List[torch.Tensor], n_terms: int = 5) -> List[List[List[Tuple[str, float]]]]:
        """
        Signifies the clusters found by the model.

        Args:
            cluster_reprs (List[torch.Tensor]): The cluster representatives.
            n_terms (int): The number of terms to use. Defaults to 5.
        
        Returns:
            List[List[List[Tuple[str, float]]]]: The cluster interpretations.
        """
        cluster_interps = []

        with torch.no_grad():
            for cluster_repr in cluster_reprs:
                cluster_interp = []
                cluster_repr = cluster_repr.to(device)

                for group in self._signifiers_groups:
                    if len(group) > 0:
                        signifiers = torch.cat([clip.tokenize(s) for _, s in group]).to(device)
                        signifiers = self._model.encode_text(signifiers)
                        signifiers = signifiers / signifiers.norm(dim=-1, keepdim=True)
                        # Compute similarity
                        similarity = 100.0 * cluster_repr @ signifiers.t()

                        values, indices = similarity.topk(min(n_terms, len(group)))
                        group_interp = [(group[i][0], v.item()) for i, v in zip(indices, values)]
                        cluster_interp.append(group_interp)
                cluster_interps.append(cluster_interp)
        return cluster_interps
    
    def _visualize_with_umap(self, method: str) -> None:
        """
        Visualizes the clusters found by the model fitting UMAP on a 20% stratified sample of the embeddings.

        Args:
            method (str): The clustering method used.

        Returns:
            None
        """
        # 2D visualization
        reducer_2d = UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.0,
            spread=1.5,
            metric="cosine",
            random_state=42
        )
        embeddings_2d = reducer_2d.fit_transform(self._embeddings)
        sample = train_test_split(embeddings_2d, self._labels, test_size=.8, stratify=self._labels, random_state=42)
        sampled_embeddings, sampled_labels = sample[0], sample[2]

        plt.figure(figsize=(10, 7))
        plt.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1], c=sampled_labels, cmap="viridis", s=4, alpha=.7)
        plt.title(f"Clusters found by {method.upper()} visualized with UMAP")
        plt.colorbar()
        plt.savefig(f"results/{method}.svg", format="svg", bbox_inches="tight")
        plt.close()
