"""
Classes and functions for topic modeling of artworks.
"""

import warnings
import argparse
import pickle
import glob
import clip
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from sklearn.cluster import KMeans, DBSCAN
from typing import Tuple, List, Dict
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from umap import UMAP
from PIL import Image
import pandas as pd
import numpy as np
import hdbscan

from CLIPFinetuning import ImageCaptionDataset, load_model
from LLMExplaining import Explainer

# General settings
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


class TopicModeler:

    def __init__(self,
                 embedding_model: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt"),
                 embeddings_path: str = "data/finetuned_embeddings.csv",
                 vocabulary_path: str = "data/topic_vocab.pkl",
                 theme_names: List[str] = ["Genre", "Subject", "Medium", "Style"],
                 min_topic_size: int = 20,
                 top_n_words: int = 5,
                 nr_topics: int = 10) -> None:
        """
        Initializes the TopicModeler.
        
        Args:
            embedding_model (Tuple[str, str]): The embedding model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
            embeddings_path (str): The path to the embeddings csv. Defaults to "data/finetuned_embeddings.csv".
            vocabulary_path (str): The path to the vocabulary pickle. Defaults to "data/topic_vocab.pkl".
            theme_names (List[str]): The theme names. Defaults to ["Genre", "Subject", "Medium", "Style"].
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

        # Loading the topic vocabulary
        self._topic_words, self._topic_phrases = pickle.load(open(vocabulary_path, "rb"))
        self._min_topic_size = min_topic_size
        self._top_n_words = top_n_words
        self._nr_topics = nr_topics

        self._theme_names = theme_names
        self._labels, self._probs = None, None
        self._centers, self._topics = [], []
        self._scores = {}

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
        if method == "hdbscan":
            self._probs = self._cluster_model[method].probabilities_
        self._compute_centers()
        print(f"Found {self._nr_topics} topics!\n")

        # Extracting the topics
        self._extract_topics()
        self._evaluate_topics()

        print(f"Topic diversity: {self._scores['topic_diversity']:.4f}")
        print(f"Silhouette score: {self._scores['silhouette']:.4f}")
        # Saving & visualizing the results
        self._save_results()
        self._visualize_topics()
        self._visualize_embeddings()
    
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
        with torch.no_grad():
            for words, phrases in zip(self._topic_words, self._topic_phrases):
                phrases = clip.tokenize([p.lower() for p in phrases]).to(device)
                phrases = self._embedding_model.encode_text(phrases)
                phrases = phrases / phrases.norm(dim=-1, keepdim=True)

                theme = []
                # Computing the cosine similarity with each cluster center
                for center in self._centers:
                    center = torch.from_numpy(center).float().to(device)
                    similarity = center @ phrases.t()

                    values, indices = similarity.topk(self._top_n_words)
                    topic = [(words[i], v.item()) for i, v in zip(indices, values)]
                    theme.append(topic)

                self._topics.append(theme)

    def _evaluate_topics(self) -> None:
        """
        Evaluates the topics in terms of diversity.

        Returns:
            None
        """
        topk_words = self._top_n_words * len(self._topics)
        diversity = TopicDiversity(topk=topk_words)
        topics = []
        # Merging the themes for each topic
        for i in range(self._nr_topics):
            topic = []
            for j in range(len(self._topics)):
                theme = [w for w, _ in self._topics[j][i]]
                topic.extend(theme)
            topics.append(topic)
        
        output_tm = {"topics": topics}
        score = diversity.score(output_tm)
        self._scores["topic_diversity"] = score

    def _save_results(self) -> None:
        """
        Saves the results.

        Returns:
            None
        """
        saving = {
            "topics": self._topics,
            "scores": self._scores
        }
        with open(f"results/topic_modeling.pkl", "wb") as f:
            pickle.dump(saving, f)
    
    def _visualize_topics(self) -> None:
        """
        Visualizes a sample for each topic.

        Returns:
            None
        """
        self._embeddings_df["label"] = self._labels
        
        for label, df in self._embeddings_df.groupby("label"):
            sample = df["image_path"].sample(20, random_state=0)
            fig, axes = plt.subplots(4, 5, figsize=(20, 27))
            axes = axes.flatten()
            for ax, image_path in zip(axes, sample):
                ax.imshow(Image.open(image_path).convert("RGB"))
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"results/sample{label+1:02d}.png", format="png", dpi=300, bbox_inches="tight")
            # Adding the topic table
            topic = [t[label] for t in self._topics]
            headers = [n.capitalize() for n in self._theme_names]
            table = [[f"{t.upper()} ({v:.2f})" for t, v in theme] for theme in topic]
            table = list(map(list, zip(*table)))

            table_ax = fig.add_subplot(111, frame_on=False)
            table_ax.axis("off")
            table_plot = table_ax.table(
                cellText=table,
                colLabels=headers,
                cellLoc="center",
                loc="bottom",
                colColours=["lightgray"] * len(headers),
                bbox=[0, -.2, 1, .23]
            )
            table_plot.auto_set_font_size(False)
            table_plot.set_fontsize(19)

            plt.tight_layout()
            plt.savefig(f"results/topic{label+1:02d}.png", format="png", dpi=300, bbox_inches="tight")
            plt.close()

    def _visualize_embeddings(self) -> None:
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
        # Considering a sample for visualization
        sample = train_test_split(embeddings_2d, self._labels, train_size=.01, stratify=self._labels, random_state=42)
        sampled_embeddings, sampled_labels = sample[0], sample[2]

        plt.figure(figsize=(17, 12))
        plt.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1],
                    c=sampled_labels, cmap="viridis", s=40, marker="o")
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c=np.arange(len(self._centers)),
                    cmap="viridis", s=450, marker="x")
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
    parser.add_argument("--min_topic_size", type=int, default=20)
    parser.add_argument("--top_n_words", type=int, default=5)
    parser.add_argument("--nr_topics", type=int, default=10)

    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--reduce", action="store_true")
    parser.add_argument("--explain", action="store_true")

    args = parser.parse_args()

    # 1. initialize the topic modeler
    topic_modeler = TopicModeler(
        embedding_model=args.embedding_model.split(";"),
        embeddings_path=args.embeddings_path,
        vocabulary_path=args.vocabulary_path,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        nr_topics=args.nr_topics
    )

    # 2. fit the topic modeler
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
