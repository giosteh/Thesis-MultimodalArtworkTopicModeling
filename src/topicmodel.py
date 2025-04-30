"""
Classes and functions for topic modeling of artworks.
"""

from metrics import TopicDiversity, ImageEmbeddingPairwiseSimilarity, ImageEmbeddingCoherence
from finetuneCLIP import ImageCaptionDataset, load_model
from sklearn.model_selection import train_test_split
from scipy.sparse import SparseEfficiencyWarning
from numba.core.errors import NumbaWarning
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from typing import Tuple, List
from fcmeans import FCM
from umap import UMAP
from PIL import Image
import pandas as pd
import numpy as np
import warnings
import pickle
import torch
import clip
import os

plt.rcParams["font.family"] = "Ubuntu Mono"
# Warnings handling
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


device = "cuda" if torch.cuda.is_available() else "cpu"



class EmbeddingDatasetBuilder:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = "models/finetuned-v2.pt",
                 paths_dataset: ImageCaptionDataset = ImageCaptionDataset(path_only=True),
                 dataset: ImageCaptionDataset = ImageCaptionDataset()):
        """Initializes the EmbeddingDatasetBuilder.
        
        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to "models/finetuned.pt".
            paths_dataset (ImageCaptionDataset): The paths dataset. Defaults to ImageCaptionDataset(path_only=True).
            dataset (ImageCaptionDataset): The dataset. Defaults to ImageCaptionDataset().
        """
        self._finetuned_model = load_model(base_model, model_path)
        # Defining the dataloaders
        self._paths_loader = DataLoader(paths_dataset, batch_size=1, shuffle=False)
        self._data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


    def __call__(self) -> pd.DataFrame:
        """Builds the embedding dataset to use for topic modeling.
        
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
                 encoder: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt"),
                 embeddings_path: str = "data/finetuned_embeddings.csv",
                 vocabulary_path: str = "data/topic_vocab.pkl",
                 pov_names: List[str] = ["Genre", "Subject", "Medium", "Style"],
                 min_topic_size: int = 100,
                 top_n_images: int = 20,
                 top_n_words: int = 10,
                 nr_topics: int = 10,
                 reduced_dim: int = 5):
        """Initializes the TopicModel.
        
        Args:
            encoder (Tuple[str, str]): The encoder model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
            embeddings_path (str): The path to the embeddings csv. Defaults to "data/finetuned_embeddings.csv".
            vocabulary_path (str): The path to the vocabulary pickle. Defaults to "data/topic_vocab.pkl".
            pov_names (List[str]): The pov names. Defaults to ["Genre", "Subject", "Medium", "Style"].
            min_topic_size (int): The minimum topic size. Defaults to 20.
            top_n_images (int): The number of top images to use. Defaults to 20.
            top_n_words (int): The number of top words to use. Defaults to 10.
            nr_topics (int): The number of topics to find. Defaults to 10.
            reduced_dim (int): The dimension to reduce the embeddings to. Defaults to 5.
        """
        self._encoder = load_model(encoder[0], encoder[1])
        # Loading the embeddings
        self.df = pd.read_csv(embeddings_path)
        embeddings = self.df["embedding"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        embeddings = np.vstack(embeddings.values)
        # Normalizing the embeddings
        X_torch = torch.from_numpy(embeddings).float()
        X_normalized = X_torch / X_torch.norm(dim=-1, keepdim=True)
        self._embeddings = X_normalized.cpu().numpy()

        self._topic_words, self._topic_prompts = pickle.load(open(vocabulary_path, "rb"))
        self._min_topic_size = min_topic_size
        self._top_n_images = top_n_images
        self._top_n_words = top_n_words
        self._nr_topics = nr_topics
        self._pov_names = pov_names

        self._centers, self._scores = [], {}
        self._topics, self._image_topics = [], []
        self._labels, self._probs = [], []
        self._images_top = []

        self.output_dir = None
        self._method = None
        # Initializing the clustering models
        self._model = {
            "kmeans": KMeans(
                n_clusters=nr_topics,
                init="k-means++",
                n_init=10,
                max_iter=5000,
                random_state=42
            ),
            "fcmeans": FCM(
                n_clusters=nr_topics,
                max_iter=500,
                random_state=42
            ),
            "birch": Birch(
                n_clusters=nr_topics,
                threshold=0.3,
                branching_factor=150
            )
        }
        # Initializing the dimensionality reduction model
        self._umap_model = UMAP(
            n_components=reduced_dim,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )


    def fit(self, method: str = "kmeans", reduce: bool = True) -> None:
        """Fits the topic modeler.
        
        Args:
            method (str): The clustering method to use. Defaults to "kmeans".
            reduce (bool): Whether to reduce the embeddings. Defaults to True.
        
        Returns:
            None
        """
        method = method.lower()
        self._method = method
        self.output_dir = f"output/{method}{self._nr_topics}"
        self.output_dir = f"{self.output_dir}R" if reduce else self.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Resetting the results
        self._centers, self._scores = [], {}
        self._topics, self._image_topics = [], []
        self._labels, self._probs = [], []
        self._images_top = []

        # Fitting the umap model
        if reduce:
            embeddings = self._umap_model.fit_transform(self._embeddings)
        else:
            embeddings = self._embeddings
        
        # Fitting the chosen clustering model
        self._model[method].fit(embeddings)
        if method == "fcmeans":
            self._labels = self._model[method].predict(embeddings)
        else:
            self._labels = self._model[method].labels_
        self._probs = getattr(self._model[method], "probabilities_", None)
        self._scores["Inertia"] = getattr(self._model[method], "inertia_", None)
        # Computing the cluster centers
        self._compute_centers()

        self.df["label"] = self._labels
        # Extracting word and image topics
        self._extract_topics()
        self._extract_image_topics()
        self._evaluate()

        self._view_topics()
        self._view_latent_space()
        self._save_results()

        return self._topics, self._image_topics, self._images_top, self._scores

    def _compute_centers(self) -> None:
        """Computes the cluster centers according to the clustering method.

        Returns:
            None
        """
        unique_labels = np.unique(self._labels)
        weighted = self._probs is not None
        
        # Computing the cluster centers
        for label in unique_labels:
            if label == -1:
                continue
            mask = self._labels == label
            center = self._embeddings[mask].mean(axis=0)
            if weighted:
                center = np.average(self._embeddings[mask], weights=self._probs[mask], axis=0)
            # Normalizing the center
            center = torch.from_numpy(center).float().to(device)
            center = center / center.norm(dim=-1, keepdim=True)
            self._centers.append(center.cpu().numpy())

    def _extract_topics(self) -> None:
        """Extracts the topics from the cluster centers.

        Returns:
            None
        """
        topics = []
        with torch.no_grad():
            for words, prompts in zip(self._topic_words, self._topic_prompts):
                prompts = clip.tokenize([s.lower() for s in prompts]).to(device)
                prompts = self._encoder.encode_text(prompts)
                prompts = prompts / prompts.norm(dim=-1, keepdim=True)

                pov = []
                # Computing the cosine similarity with each cluster center
                for center in self._centers:
                    center = torch.from_numpy(center).float().to(device)
                    similarity = center @ prompts.t()
                    # Selecting the most representative words
                    top_n = self._top_n_words // len(self._pov_names)
                    values, indices = similarity.topk(top_n)
                    topic = [(words[i], v.item()) for i, v in zip(indices, values)]
                    pov.append(topic)
                topics.append(pov)
        # Rearranging each topic povs into a single list
        self._topics = [
            [word for pov in topics for word, _ in pov[i]]
            for i in range(self._nr_topics)
        ]

    def _evaluate(self) -> None:
        """Evaluates the topics computing diversity and coherence metrics.

        Returns:
            None
        """
        topk_words = self._top_n_words * len(self._pov_names)
        topk_images = self._top_n_images
        metrics = {
            "TD": TopicDiversity(topk=topk_words),
            "IEPS": ImageEmbeddingPairwiseSimilarity(topk=topk_images),
            "IEC": ImageEmbeddingCoherence(topk=topk_images)
        }
        # Computing the metrics
        for metric_name, metric in metrics.items():
            topics = self._topics if metric_name not in ["IEPS", "IEC"] else self._image_topics
            score = metric(topics)
            self._scores[metric_name] = score
            print(f"{metric_name}: {score:.4f}")

    def _save_results(self) -> None:
        """Saves the results.

        Returns:
            None
        """
        saving = {"topics": self._topics, "scores": self._scores}
        with open(f"{self.output_dir}/results.pkl", "wb") as f:
            pickle.dump(saving, f)
    
    def _extract_image_topics(self) -> None:
        """Extracts the top images for each topic based on the similarity with the cluster centers.
        
        Returns:
            None
        """
        centers = np.array(self._centers)
        images_similarity = self._embeddings @ centers.T
        row_idx = np.arange(len(images_similarity))
        # Adding the similarity column for each image
        self.df["similarity"] = images_similarity[row_idx, self.df["label"]]

        self._image_topics = (
            self.df.groupby("label", group_keys=False)
            .apply(lambda x: x.nlargest(self._top_n_images, "similarity")["image_path"].tolist(), include_groups=False)
            .tolist()
        )

    def _view_topics(self) -> None:
        """Visualizes the top and sampled images for each topic.

        Returns:
            None
        """
        # Iterating over the topics
        for label, df in self.df.groupby("label"):
            if label == -1:
                continue
            topic, image_topic = self._topics[label], self._image_topics[label]
            image_sample = df["image_path"].sample(min(self._top_n_images, len(df)), random_state=0).tolist()

            self._view_single_topic(f"{self.output_dir}/topic{label+1}T.png", image_topic, topic)
            self._view_single_topic(f"{self.output_dir}/image{label+1}T.png", image_topic)
            self._images_top.append(f"{self.output_dir}/image{label+1}T.png")

    def _view_single_topic(self,
                           path: str,
                           images: List[str],
                           topic: List[str] = None) -> None:
        """Visualizes a single topic.

        Args:
            path (str): The path to save the figure.
            images (List[str]): The images to show.
            topic (List[str], optional): The topic to show. Defaults to None.

        Returns:
            None
        """
        fig, axes = plt.subplots(4, 5, figsize=(20, 24))
        axes = axes.flatten()
        for ax, image_path in zip(axes, images):
            ax.imshow(Image.open(image_path).convert("RGB"))
            ax.axis("off")
        plt.tight_layout()

        if topic is None:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")
            plt.close()
            return
        # Adding the text
        fig.subplots_adjust(bottom=.2)
        text_ax = fig.add_axes([.05, .07, .9, .12])
        povs = np.array_split(topic, len(self._pov_names))
        text_content = "\n\n".join(
            f"{pov_name.upper()} : " + ", ".join(povs[i])
            for i, pov_name in enumerate(self._pov_names)
        )
        text_ax.text(0, 1, text_content, fontsize=26, ha="left", va="top",
                     bbox=dict(boxstyle="square,pad=1.2", facecolor="#f6f6f6"))
        text_ax.axis("off")
        plt.savefig(path, format="png", dpi=300, bbox_inches="tight")
        plt.close()

    def _view_latent_space(self) -> None:
        """Visualizes the embeddings in a 2D space.

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

        plt.figure(figsize=(15, 12))
        plt.tight_layout()
        plt.scatter(sampled_embeddings[:, 0], sampled_embeddings[:, 1], c=sampled_labels,
                    cmap="viridis", s=30, marker="o", alpha=.75)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c=np.arange(len(self._topics)), cmap="viridis", s=680, marker="*")
        plt.title("Latent Space in 2D")
        plt.savefig(f"{self.output_dir}/latentspace.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()
