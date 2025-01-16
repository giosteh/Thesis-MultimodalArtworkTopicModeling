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
from cluster_explaining import Explainer

from typing import List, Dict
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import glob
import argparse
import warnings

# Setting some things up
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"font.family": "DejaVu Sans"})


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

                features["image_path"] = image_path.strip(",'()")
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
        # Loading the dataset
        self._df = pd.read_csv(dataset_path)
        embeddings = self._df["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        embeddings = np.vstack(embeddings.values)
        # Normalizing
        X_torch = torch.from_numpy(embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        self._embeddings = X_normalized.cpu().numpy()

        self._clusterer = None
        self.labels = None
        self.centers = []
        self.interps = []
        # Loading signifiers for cluster interpretation
        with open(signifiers_path, "rb") as f:
            self.signifiers = pickle.load(f) # Must be a tuple holding two lists
    

    def cluster(self, method: str = "kmeans", n_terms: int = 5, **kwds) -> None:
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
            case "kmedoids":
                self._clusterer = KMedoids(
                    n_clusters=kwds["n_clusters"] if "n_clusters" in kwds else 10,
                    init="k-medoids++",
                    metric="cosine",
                    method="pam",
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
        self.labels = self._clusterer.fit_predict(self._embeddings)
        self._get_cluster_centers()
        self._signify_clusters(n_terms=n_terms)

        # Saving stats and interps
        n_clusters = len(self.centers)
        path = f"results/{method}{n_clusters:02d}"
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "stats": self._stats(),
                "interps": self.interps,
                "interps_stats": self._interps_stats()
            }, f)
        # Visualizing
        self._visualize_embedding_space(path)
        self._visualize_clusters(path, n_samples=20)

    def _stats(self) -> Dict[str, float]:
        """
        Gets the clustering statistics.

        Returns:
            Dict[str, float]: The clustering statistics.
        """
        return {
            "labels": self.labels,
            "sizes": np.bincount(self.labels[self.labels != -1]),
            "silhouette": silhouette_score(self._embeddings, self.labels),
            "calinski_harabasz": calinski_harabasz_score(self._embeddings, self.labels),
            "inertia": self._clusterer.inertia_ if hasattr(self._clusterer, "inertia_") else None
        }
    
    def _interps_stats(self) -> Dict[str, List[float]]:
        """
        Gets the interpretation statistics, i.e. the average overlap between clusters.

        Returns:
            Dict[str, List[float]]: The interpretation statistics.
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
            group_interp = self.interps[cluster_idx][group_idx]
            group_terms, n_terms = set([t for t, _ in group_interp]), len(group_interp)
            overlap = 0
            for i, interp in enumerate(self.interps):
                if i == cluster_idx:
                    continue
                same_group_terms = set([t for t, _ in interp[group_idx]])
                overlap += len(same_group_terms.intersection(group_terms)) / n_terms
            return overlap / (n_clusters - 1)
        
        stats = {
            "avg_overlap_per_group": [],
            "avg_overlap_per_cluster": []
        }
        n_clusters, n_groups = len(self.centers), len(self.signifiers[0])
        overlaps = np.zeros((n_clusters, n_groups))
        for i in range(n_clusters):
            for j in range(n_groups):
                overlaps[i, j] = avg_overlap(j, i)
        
        stats["avg_overlap_per_group"] = np.mean(overlaps, axis=0).tolist()
        stats["avg_overlap_per_cluster"] = np.mean(overlaps, axis=1).tolist()
        return stats

    def _get_cluster_centers(self) -> None:
        """
        Gets the centers for each of the clusters found.

        Returns:
            None
        """
        # Computing the centers
        unique_labels = np.unique(self.labels[self.labels != -1])
        self.centers = [self._embeddings[self.labels == label].mean(axis=0) for label in unique_labels]
    
    def _signify_clusters(self, n_terms: int = 5) -> None:
        """
        Signifies the clusters found using the cluster centers.

        Args:
            n_terms (int): The number of terms to use. Defaults to 5.

        Returns:
            None
        """
        with torch.no_grad():
            for center in self.centers:
                interp = []
                center = torch.from_numpy(center).float().to(device)
                # Iterating over the groups
                for group in self.signifiers[1]:
                    signifiers = clip.tokenize([s.lower() for _, s in group]).to(device)
                    signifiers = self._model.encode_text(signifiers)
                    signifiers = signifiers / signifiers.norm(dim=-1, keepdim=True)
                    # Computing the cosine similarity
                    similarity = center @ signifiers.t()

                    values, indices = similarity.topk(min(n_terms, len(group)))
                    group_interp = [(group[i][0], v.item()) for i, v in zip(indices, values)]
                    interp.append(group_interp)

                self.interps.append(interp)
    
    def _visualize_clusters(self, path: str, n_samples: int) -> None:
        """
        Visualizes a sample of the clusters found.

        Args:
            path (str): The path to save the image.
            n_samples (int): The number of samples to plot.

        Returns:
            None
        """
        self._df["cluster"] = self.labels

        for cluster_label, cluster_df in self._df.groupby("cluster", sort=True):
            sample_images = cluster_df["image_path"].sample(n_samples, random_state=0)
            # Plotting & saving
            fig, axes = plt.subplots(n_samples // 5, 5, figsize=(20, 27))
            axes = axes.flatten()
            for ax, image_path in zip(axes, sample_images):
                ax.imshow(Image.open(image_path.strip(",'()")).convert("RGB"))
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"{path}_cluster{cluster_label+1:02d}.png", format="png", dpi=300, bbox_inches="tight")

            # Saving the samples along with the cluster description
            interp = self.interps[cluster_label]
            headers = [g.capitalize() for g in self.signifiers[0]]
            table = [[f"{t.lower()} ({v:.2f})" for t, v in group] for group in interp]
            table = list(map(list, zip(*table)))

            table_ax = fig.add_subplot(111, frame_on=False)
            table_ax.axis("off")
            table_plot = table_ax.table(cellText=table, colLabels=headers, loc="bottom",
                                        cellLoc="center", colColours=["lightgray"] * len(headers),
                                        bbox=[0, -.2, 1, .23])
            table_plot.auto_set_font_size(False)
            table_plot.set_fontsize(19)
            
            plt.tight_layout()
            plt.savefig(f"{path}_interp{cluster_label+1:02d}.png", format="png", dpi=300, bbox_inches="tight")
            plt.close()
    
    def _visualize_embedding_space(self, path: str) -> None:
        """
        Visualizes the embedding space.

        Args:
            path (str): The path to save the image.

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
        centers = reducer.transform(self.centers)
        
        sample = train_test_split(embeddings, self.labels, train_size=1000, stratify=self.labels, random_state=42)
        sampled_embeddings, sampled_labels = sample[0], sample[2]
        # Plotting & saving
        plt.figure(figsize=(15, 12))
        plt.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1],
                    c=sampled_labels, cmap="viridis", s=40, alpha=.7, marker="h")
        plt.scatter(centers[:, 0], centers[:, 1],
                    c=np.arange(len(self.centers)), cmap="viridis", s=350, marker="x")
        plt.colorbar()
        
        plt.title("Artwork Embedding Space")
        plt.savefig(f"{path}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--finetuned_model", type=str, default="models/finetuned-v2.pt")
    parser.add_argument("--dataset", type=str, default="data/finetuned_embeddings.csv")
    parser.add_argument("--signifiers", type=str, default="data/signifiers.pkl")
    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--n_terms", type=int, default=5)
    parser.add_argument("--explain", action="store_true")

    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--min_samples", type=int, default=128)

    args = parser.parse_args()

    # 1. initialize the clusterer
    clusterer = ArtworkClusterer(
        model_path=args.finetuned_model,
        dataset_path=args.dataset,
        signifiers_path=args.signifiers
    )

    # 2. perform clustering
    clusterer.cluster(
        method=args.method,
        n_terms=args.n_terms,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples
    )

    # 3. explain the clusters
    if args.explain:
        with open(f"results/{args.method}{args.n_clusters:02d}.pkl", "rb") as f:
            interps = pickle.load(f)["interps"]
        image_paths = sorted(glob.glob(f"results/{args.method}{args.n_clusters:02d}_cluster*.png"))

        explainer = Explainer(
            model_path=args.finetuned_model
        )
        # 3.1 explain with images only
        explainer(image_paths, interps, path=f"results/descr{args.n_clusters:02d}_images_only")
        # 3.2 explain with images and terms
        explainer(image_paths, interps, comprehensive=True, path=f"results/descr{args.n_clusters:02d}_with_terms")
