"""
Classes and functions for clustering artworks.
"""

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans, DBSCAN

from clip_finetuning import ImageCaptionDataset

from typing import List
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
                 finetuned_model_path: str = "models/checkpoint.pt",
                 raw_dataset: ImageCaptionDataset = ImageCaptionDataset(raw_only=True),
                 dataset: ImageCaptionDataset = ImageCaptionDataset()) -> None:
        """
        Initializes the EmbeddingDatasetBuilder.
        """
        self._base_model, _ = clip.load(base_model, device=device, jit=False)
        self._base_model.float()
        self._base_model.eval()
        self._finetuned_model = load_finetuned_model(finetuned_model_path)

        self._raw_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False)
        self._data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    

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

        return pd.DataFrame(rows)


class ArtworkClusterer:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 finetuned_model_path: str = None,
                 dataset: pd.DataFrame = EmbeddingDatasetBuilder()(),
                 vocabulary_file: List[str] = "data/vocabulary.pkl") -> None:
        """
        Initializes the ArtworkClusterer.
        """
        self._model, _ = clip.load(base_model, device=device, jit=False)
        self._model.float()
        self._model.eval()
        if finetuned_model_path:
            self._model = load_finetuned_model(finetuned_model_path)

        dataset["embedding"] = dataset["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        self._embeddings = np.vstack(dataset["embedding"].values)

        with open(vocabulary_file, "rb") as f:
            self._vocabulary = pickle.load(f)
    

    def signify_clusters(self, centroids: List[torch.Tensor], n_labels: int = 10) -> List[List[(str, float)]]:
        """
        Signifies the clusters assigning the most similar labels to each centroid.

        Args:
            centroids (List[torch.Tensor]): The centroids.
            n_labels (int): The number of labels to assign.

        Returns:
            List[List[(str, float)]]: The list of labels for each centroid.
        """
        cluster_interpretations = []
        signifiers = torch.cat([clip.tokenize(s) for s in self._vocabulary]).to(device)

        with torch.no_grad():
            for centroid in centroids:
                centroid = centroid.to(device)
                signifiers = self._model.encode_text(signifiers)
                signifiers = signifiers / signifiers.norm(dim=-1, keepdim=True)

                similarity = (100 * centroid @ signifiers.t()).softmax(dim=-1)
                values, indices = similarity[0].topk(n_labels)
                interpretation = [(self._vocabulary[i], v.item()) for i, v in zip(indices, values)]

                cluster_interpretations.append(interpretation)

        return cluster_interpretations

    def cluster_with_kmeans(self, n_clusters: int = 10) -> List[torch.Tensor]:
        """
        Clusters the embeddings using k-means.

        Args:
            n_clusters (int): The number of clusters to use.

        Returns:
            List[torch.Tensor]: The cluster centers.
        """
        X_torch = torch.from_numpy(self._embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        X = X_normalized.cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_

        return [torch.from_numpy(c).float() for c in centroids]
    
    def cluster_with_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> List[torch.Tensor]:
        pass