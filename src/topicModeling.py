"""
Classes and functions for topic modeling of artworks.
"""

from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from CLIPFinetuning import ImageCaptionDataset, load_model
from scipy.sparse import SparseEfficiencyWarning
from numba.core.errors import NumbaWarning
from sklearn.cluster import KMeans, DBSCAN
from LLMExplaining import Explainer
import matplotlib.pyplot as plt
from typing import Tuple, List
from umap import UMAP
from PIL import Image
import pandas as pd
import numpy as np
import hdbscan
import warnings
import argparse
import pickle
import glob
import clip
import torch


# General settings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"font.family": "Lato"})


device = "cuda" if torch.cuda.is_available() else "cpu"



class EmbeddingDatasetBuilder:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = "models/finetuned-v2.pt",
                 paths_dataset: ImageCaptionDataset = ImageCaptionDataset(path_only=True),
                 dataset: ImageCaptionDataset = ImageCaptionDataset()) -> None:
        """
        Initializes the EmbeddingDatasetBuilder.
        
        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to "models/finetuned.pt".
            paths_dataset (ImageCaptionDataset): The paths dataset. Defaults to ImageCaptionDataset(path_only=True).
            dataset (ImageCaptionDataset): The dataset. Defaults to ImageCaptionDataset().
        """
        self._finetuned_model = load_model(base_model, model_path)
        self._paths_loader = DataLoader(paths_dataset, batch_size=1, shuffle=False)
        self._data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


    def __call__(self) -> pd.DataFrame:
        """
        Builds the embedding dataset to use for topic modeling.
        
        Returns:
            pd.DataFrame: The embedding dataset.
        """
        rows = []

        with torch.no_grad():
            for paths, data in zip(self._paths_loader, self._data_loader):
                features = {}
                image_path, _ = paths
                image, text = data

                features["image_path"] = str(image_path).strip(",'()")
                features["text"] = str(text)
                image_embedding = self._finetuned_model.encode_image(image.to(device))
                image_embedding = image_embedding.cpu().numpy().flatten()
                features["embedding"] = np.array2string(image_embedding, np.inf, separator=",")

                rows.append(features)
        return pd.DataFrame(rows)


class TopicModel:

    def __init__(self,
                 embedding_model: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt"),
                 embeddings_path: str = "data/finetuned_embeddings.csv",
                 vocabulary_path: str = "data/topic_vocab.pkl",
                 pov_names: List[str] = ["Genre", "Subject", "Medium", "Style"],
                 min_topic_size: int = 20,
                 top_n_words: int = 5,
                 nr_topics: int = 10) -> None:
        """
        Initializes the TopicModel.
        
        Args:
            embedding_model (Tuple[str, str]): The embedding model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
            embeddings_path (str): The path to the embeddings csv. Defaults to "data/finetuned_embeddings.csv".
            vocabulary_path (str): The path to the vocabulary pickle. Defaults to "data/topic_vocab.pkl".
            pov_names (List[str]): The pov names. Defaults to ["Genre", "Subject", "Medium", "Style"].
            min_topic_size (int): The minimum topic size. Defaults to 20.
            top_n_words (int): The number of top words to use. Defaults to 5.
            nr_topics (int): The number of topics to find. Defaults to 10.
        """
        self._embedding_model = load_model(embedding_model[0], embedding_model[1])
        # Loading the embeddings
        self._embeddings_df = pd.read_csv(embeddings_path)
        embeddings = self._embeddings_df["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        embeddings = np.vstack(embeddings.values)
        # Normalizing the embeddings
        X_torch = torch.from_numpy(embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        self._embeddings = X_normalized.cpu().numpy()

        self._topic_words, self._topic_sentences = pickle.load(open(vocabulary_path, "rb"))
        self._min_topic_size = min_topic_size
        self._top_n_words = top_n_words
        self._nr_topics = nr_topics
        self._pov_names = pov_names

        self._labels, self._probs = None, None
        self._centers, self._topics = [], []
        self._scores = {}
        self._top_images = None

        # Initializing the cluster model
        self._cluster_model = {
            "kmeans": KMeans(
                n_clusters=nr_topics,
                init="k-means++",
                n_init=10,
                max_iter=10000,
                random_state=42
            ),
            "dbscan": DBSCAN(
                eps=.3,
                min_samples=20,
                metric="cosine"
            ),
            "hdbscan": hdbscan.HDBSCAN(
                min_cluster_size=min_topic_size,
                metric="euclidean",
                cluster_selection_method="eom"
            )
        }
        # Initializing the dimensionality reduction model
        self._umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=.0,
            metric="cosine",
            random_state=42
        )
    

    def fit(self, method: str = "kmeans", reduce: bool = True) -> None:
        """
        Fits the topic modeler.
        
        Args:
            method (str): The clustering method to use. Defaults to "kmeans".
            reduce (bool): Whether to reduce the embeddings. Defaults to True.
        
        Returns:
            None
        """
        method = method.lower()
        if reduce:
            embeddings = self._umap_model.fit_transform(self._embeddings)
        else:
            embeddings = self._embeddings
        
        # Fitting the cluster model
        self._cluster_model[method].fit(embeddings)
        self._labels = self._cluster_model[method].labels_
        self._embeddings_df["label"] = self._labels
        if method == "hdbscan":
            self._probs = self._cluster_model[method].probabilities_
        self._compute_centers()
        print(f"Found {self._nr_topics} topics!\n")
        # Percentage of noise
        noise_perc = len(self._labels[self._labels == -1]) / len(self._labels)
        print(f"Noise: {noise_perc*100:.2f}%\n")

        self._extract_topics()
        self._evaluate_topics()
        # Saving the results and visualizing
        self._save_results()
        self._view_topics()
        self._view_latent_space()
    
    def _compute_centers(self) -> None:
        """
        Computes the cluster centers according to the clustering method.

        Returns:
            None
        """
        labels_filtered = self._labels[self._labels != -1]
        embeddings_filtered = self._embeddings[self._labels != -1]

        unique_labels = np.unique(labels_filtered)
        self._nr_topics = len(unique_labels)
        weighted = True if self._probs is not None else False

        for l in unique_labels:
            mask = (self._labels == l)
            center = self._embeddings[mask].mean(axis=0)
            if weighted:
                center = np.average(self._embeddings[mask], weights=self._probs[mask], axis=0)
            self._centers.append(center)
        
        # Computing the silhouette score
        if len(unique_labels) < 2:
            self._scores["silhouette"] = None
            return
        score = silhouette_score(embeddings_filtered, labels_filtered)
        self._scores["silhouette"] = score

    def _extract_topics(self) -> None:
        """
        Extracts the topics from the cluster centers.

        Returns:
            None
        """
        topics = []
        with torch.no_grad():
            for words, sentences in zip(self._topic_words, self._topic_sentences):
                sentences = clip.tokenize([p.lower() for p in sentences]).to(device)
                sentences = self._embedding_model.encode_text(sentences)
                sentences = sentences / sentences.norm(dim=-1, keepdim=True)

                pov = []
                # Computing the cosine similarity with each cluster center
                for center in self._centers:
                    center = torch.from_numpy(center).float().to(device)
                    similarity = center @ sentences.t()
                    # Selecting the top n words
                    values, indices = similarity.topk(self._top_n_words)
                    topic = [(words[i], v.item()) for i, v in zip(indices, values)]
                    pov.append(topic)
                topics.append(pov)
        # Rearranging each topic povs into a single list
        self._topics = [
            [word for pov in topics for word, _ in pov[i]]
            for i in range(self._nr_topics)
        ]

    def _evaluate_topics(self) -> None:
        """
        Evaluates the topics computing the topic diversity.

        Returns:
            None
        """
        topk_words = self._top_n_words * len(self._pov_names)
        metrics = {
            "topic_diversity": TopicDiversity(topk=topk_words)
        }

        model_output = {"topics": self._topics}
        # Computing the metrics
        for metric_name, metric in metrics.items():
            score = metric.score(model_output)
            self._scores[metric_name] = score
            print(f"{metric_name}: {score:.4f}")

    def _save_results(self) -> None:
        """
        Saves the results.

        Returns:
            None
        """
        saving = {"topics": self._topics, "scores": self._scores}
        with open(f"results/TM.pkl", "wb") as f:
            pickle.dump(saving, f)
    
    def _get_top_images(self, nr_images: int = 20) -> None:
        """
        Gets the most representative images for each topic.
        
        Args:
            nr_images (int): The number of images to get. Defaults to 20.
        
        Returns:
            None
        """
        centers = np.array(self._centers)
        images_similarity = self._embeddings @ centers.T
        row_idx = np.arange(len(images_similarity))
        # Adding the similarity column for each image
        self._embeddings_df["similarity"] = images_similarity[row_idx, self._embeddings_df["label"]]
        df = self._embeddings_df.copy()
        # Getting the top images for each topic
        self._top_images = (
            df.groupby("label", group_keys=False)
            .apply(lambda x: x.nlargest(nr_images, "similarity")["image_path"].tolist(), include_groups=False)
            .tolist()
        )

    def _view_topics(self) -> None:
        """
        Visualizes a sample for each topic.

        Returns:
            None
        """
        self._get_top_images()
        for label in np.unique(self._labels):
            images = self._top_images[label]
            fig, axes = plt.subplots(4, 5, figsize=(20, 26))
            axes = axes.flatten()
            for ax, image_path in zip(axes, images):
                ax.imshow(Image.open(image_path).convert("RGB"))
                ax.axis("off")
            plt.tight_layout()
            # Adding a text box
            fig.subplots_adjust(bottom=.2)
            text_ax = fig.add_axes([0.05, 0.05, 0.9, 0.1])

            povs = np.array_split(self._topics[label], len(self._pov_names))
            text_content = "\n\n".join(
                f"{pov_name.upper()} : " + ", ".join(povs[i])
                for i, pov_name in enumerate(self._pov_names)
            )
            text_ax.text(0, 1, text_content, fontsize=24, ha="left", va="top",
                         bbox=dict(boxstyle="square,pad=1.2", facecolor="#f5f5f5"))
            text_ax.axis("off")
            plt.savefig(f"results/topic{label+1:02d}.png", format="png", dpi=300, bbox_inches="tight")
            plt.close()

    def _view_latent_space(self) -> None:
        """
        Visualizes the embeddings in a 2D space.

        Returns:
            None
        """
        umap_2d = UMAP(
            n_neighbors=15,
            min_dist=.01,
            metric="cosine",
            random_state=42
        )
        embeddings_2d = umap_2d.fit_transform(self._embeddings)
        centers_2d = umap_2d.transform(self._centers)
        topics_range = np.arange(self._nr_topics)
        # Considering a sample for visualization
        embeddings = embeddings_2d[self._labels != -1]
        labels = self._labels[self._labels != -1]
        sample = train_test_split(embeddings, labels, train_size=.01, stratify=labels, random_state=42)
        sampled_embeddings, sampled_labels = sample[0], sample[2]

        plt.figure(figsize=(17, 12))
        plt.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1],
                    c=sampled_labels, cmap="plasma", s=40, marker="o")
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c=topics_range, cmap="plasma", s=400, marker="x")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig("results/embeddings.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", type=str, default="ViT-B/32;models/finetuned-v2.pt")
    parser.add_argument("--embeddings_path", type=str, default="data/finetuned_embeddings.csv")
    parser.add_argument("--vocabulary_path", type=str, default="data/topic_vocab.pkl")
    parser.add_argument("--min_topic_size", type=int, default=25)
    parser.add_argument("--top_n_words", type=int, default=3)
    parser.add_argument("--nr_topics", type=int, default=10)

    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--reduce", action="store_true")
    parser.add_argument("--explain", action="store_true")

    args = parser.parse_args()

    # 1. initialize the topic model
    topic_modeler = TopicModel(
        embedding_model=args.embedding_model.split(";"),
        embeddings_path=args.embeddings_path,
        vocabulary_path=args.vocabulary_path,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        nr_topics=args.nr_topics
    )

    # 2. fit the topic model
    topic_modeler.fit(method=args.method, reduce=args.reduce)

    if args.explain:
        # 3. explain the topics
        explainer = Explainer(
            embedding_model=args.embedding_model.split(";")
        )
        with open("results/topic_modeling.pkl", "rb") as f:
            topics = pickle.load(f)["topics"]
        sample_paths = sorted(glob.glob("results/sample*.png"))

        explainer(
            sample_paths=sample_paths,
            saving_path="results/explaining.pkl",
            topics=topics
        )
        explainer(
            sample_paths=sample_paths,
            saving_path="results/explaining_rich.pkl",
            topics=topics,
            rich_prompt=True
        )
