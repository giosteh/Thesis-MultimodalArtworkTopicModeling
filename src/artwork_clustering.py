"""
Classes and functions for clustering artworks.
"""

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import MiniBatchKMeans
import umap

from clip_finetuning import ImageCaptionDataset

from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"



def load_finetuned_model(model_path: str) -> nn.Module:
    """
    Loads the finetuned model.

    Args:
        model_path (str): The path to the finetuned model.

    Returns:
        nn.Module: The finetuned model.
    """
    model, _ = clip.load(model_path, device=device, jit=False)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.float()
    model.eval()

    return model


class EmbeddingDatasetBuilder:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 finetuned_model_path: str = "models/finetuned.pt",
                 raw_dataset: ImageCaptionDataset = ImageCaptionDataset(raw_only=True),
                 dataset: ImageCaptionDataset = ImageCaptionDataset(),
                 use_base_model: bool = False) -> None:
        """
        Initializes the EmbeddingDatasetBuilder.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            finetuned_model_path (str): The path to the finetuned model. Defaults to "models/finetuned.pt".
            raw_dataset (ImageCaptionDataset): The raw dataset. Defaults to ImageCaptionDataset(raw_only=True).
            dataset (ImageCaptionDataset): The dataset. Defaults to ImageCaptionDataset().
            use_base_model (bool): Whether to select the base embeddings. Defaults to False.
        """
        self._base_model, _ = clip.load(base_model, device=device, jit=False)
        self._base_model.float()
        self._base_model.eval()
        self._finetuned_model = load_finetuned_model(finetuned_model_path)

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
                 finetuned_model_path: str = None,
                 dataset: pd.DataFrame = EmbeddingDatasetBuilder()(),
                 signifiers_path: str = "data/signifiers.pkl") -> None:
        """
        Initializes the ArtworkClusterer.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            finetuned_model_path (str): The path to the finetuned model. Defaults to None.
            dataset (pd.DataFrame): The dataset. Defaults to EmbeddingDatasetBuilder()().
            signifiers_path (str): The path to the signifiers. Defaults to "data/signifiers.pkl".
        """
        self._model, _ = clip.load(base_model, device=device, jit=False)
        self._model.float()
        self._model.eval()
        if finetuned_model_path:
            self._model = load_finetuned_model(finetuned_model_path)
        
        embeddings = dataset["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        self._embeddings = np.vstack(embeddings.values)

        with open(signifiers_path, "rb") as f:
            self._signifiers = pickle.load(f)
    

    def cluster(self, mode: str = "kmeans", n_clusters: int = 10, n_terms: int = 10) -> None:
        """
        Clusters the embeddings using the specified mode.

        Args:
            mode (str): The clustering mode. Defaults to "kmeans".
            n_clusters (int): The number of clusters to use. Defaults to 10.
            n_terms (int): The number of terms to assign to each cluster. Defaults to 10.
        """
        representatives, labels = None, None
        match mode:
            case "kmeans":
                representatives, labels = self._cluster_with_kmeans(n_clusters)
        
        self._visualize_with_umap(labels)
        interpretations = self._signify_clusters(representatives, n_terms)

        # Saving the interpretations
        with open("data/cluster_interpretations.pkl", "wb") as f:
            pickle.dump(interpretations, f)
    
    def _signify_clusters(self, representatives: List[torch.Tensor], n_terms: int = 10) -> List[List[(str, float)]]:
        """
        Signifies the clusters assigning the most similar labels to each centroid.

        Args:
            representatives (List[torch.Tensor]): The cluster representatives.
            n_labels (int): The number of labels to assign.

        Returns:
            List[List[(str, float)]]: The list of labels for each centroid.
        """
        cluster_interpretations = []
        signifiers = torch.cat([clip.tokenize(f"a {s} painting") for s in self._signifiers]).to(device)

        with torch.no_grad():
            for representative in representatives:
                representative = representative.to(device)
                signifiers = self._model.encode_text(signifiers)
                signifiers = signifiers / signifiers.norm(dim=-1, keepdim=True)

                similarity = (100 * representative @ signifiers.t()).softmax(dim=-1)
                values, indices = similarity[0].topk(n_terms)
                interpretation = [(self._signifiers[i], v.item()) for i, v in zip(indices, values)]

                cluster_interpretations.append(interpretation)

        return cluster_interpretations

    def _visualize_with_umap(self, labels: np.ndarray, n_neighbors: int = 30, min_dist: float = .1) -> None:
        """
        Visualizes the clusters using UMAP.

        Args:
            labels (np.ndarray): The labels.
            n_neighbors (int): The number of neighbors to use for UMAP.
            min_dist (float): The minimum distance to use for UMAP.

        Returns:
            None
        """
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embeddings_2d = reducer.fit_transform(self._embeddings)

        plt.figure(figsize=(10, 7))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="Spectral", s=5)
        plt.colorbar(label="Cluster label")
        plt.title("UMAP projection of the clustered embeddings")
        plt.savefig("umap-proj.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _cluster_with_kmeans(self, n_clusters: int = 10) -> Tuple[List[torch.Tensor], np.ndarray]:
        """
        Clusters the embeddings using KMeans.

        Args:
            n_clusters (int): The number of clusters to use.

        Returns:
            Tuple[List[torch.Tensor], np.ndarray]: The centroids and the labels.
        """
        X_torch = torch.from_numpy(self._embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        X = X_normalized.cpu().numpy()

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, init="k-means++", max_iter=1000, n_init=10, random_state=0)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        return [torch.from_numpy(c) for c in centroids], labels
