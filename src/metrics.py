"""
Classes and functions for topic modeling evaluation metrics.
"""

from sklearn.metrics import pairwise_distances
from finetuneCLIP import load_model
from abc import ABC, abstractmethod
from itertools import combinations
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"



class Metric(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, topics) -> float:
        pass


class TopicDiversity(Metric):

    def __init__(self, topk: int = 10):
        """Initializes the TopicDiversity metric.

        Args:
            topk (int): The number of top words to consider. Defaults to 10.
        """
        super().__init__()
        self.topk = topk

    def __call__(self, topics: List[List[str]]) -> float:
        """Computes the TopicDiversity metric.

        Args:
            topics (List[List[str]]): The topics to evaluate.

        Returns:
            float: The computed metric.
        """
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception("Words in topics are less than topk.")
        
        unique_words = set()
        for topic in topics:
            unique_words = unique_words.union(set(topic[:self.topk]))
        result = len(unique_words) / (self.topk * len(topics))
        return result


class ImageEmbeddingCoherence(Metric):

    def __init__(self, topk: int = 10, encoder: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt")):
        """Initializes the ImageEmbeddingCoherence metric.

        Args:
            topk (int, optional): The number of top words to consider. Defaults to 10.
            encoder (Tuple[str, str], optional): The encoder model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
        """
        super().__init__()
        self.topk = topk
        self._encoder, self._preprocess = load_model(encoder[0], encoder[1], return_preprocess=True)

    def __call__(self, topics: List[List[str]]) -> float:
        """Computes the ImageEmbeddingCoherence metric.

        Args:
            topics (List[List[str]]): The topics to evaluate.

        Returns:
            float: The computed metric.
        """
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception("Images in topics are less than topk.")
        
        result = 0.0
        for topic in topics:
            images = [self._preprocess(Image.open(path).convert("RGB")) for path in topic[:self.topk]]
            images = torch.stack(images).to(device)
            with torch.no_grad():
                E = self._encoder.encode_image(images)
            E = E / E.norm(dim=-1, keepdim=True)
            E = E.cpu().numpy()
            # Performing cosine similarity between embeddings
            similarity = np.sum(1 - pairwise_distances(E, metric="cosine") - np.diag(np.ones(self.topk)))
            coherence = similarity / (self.topk * (self.topk - 1))
            result += coherence
        # Averaging the coherence
        result /= len(topics)
        return result


class ImageEmbeddingPairwiseSimilarity(Metric):

    def __init__(self, topk: int = 10, encoder: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt")):
        """Initializes the ImageEmbeddingPairwiseSimilarity metric.

        Args:
            topk (int, optional): The number of top words to consider. Defaults to 10.
            encoder (Tuple[str, str], optional): The encoder model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
        """
        super().__init__()
        self.topk = topk
        self._encoder, self._preprocess = load_model(encoder[0], encoder[1], return_preprocess=True)

    def __call__(self, topics: List[List[str]]) -> float:
        """Computes the ImageEmbeddingPairwiseSimilarity metric.

        Args:
            topics (List[List[str]]): The topics to evaluate.

        Returns:
            float: The computed metric.
        """
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception("Images in topics are less than topk.")
        
        result = 0.0
        Es = []
        for topic in topics:
            images = [self._preprocess(Image.open(path).convert("RGB")) for path in topic[:self.topk]]
            images = torch.stack(images).to(device)
            with torch.no_grad():
                E = self._encoder.encode_image(images)
            E = E / E.norm(dim=-1, keepdim=True)
            E = E.cpu().numpy()
            Es.append(E)
        # Performing cosine similarity between each combination of topics
        topic_combinations = combinations(range(len(topics)), 2)
        for i, j in topic_combinations:
            similarity = np.sum(1 - pairwise_distances(Es[i], Es[j], metric="cosine"))
            avg_similarity = similarity / (self.topk * self.topk)
            result += avg_similarity
        # Averaging the similarity
        result /= (len(topics) * (len(topics) - 1) / 2)
        return result


class DescriptionEmbeddingSimilarity(Metric):

    def __init__(self, encoder: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt")):
        """Initializes the DescriptionEmbeddingSimilarity metric.

        Args:
            encoder (Tuple[str, str], optional): The encoder model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
        """
        super().__init__()
        self._encoder = load_model(encoder[0], encoder[1])

    def __call__(self, topics: List[str]) -> float:
        """Computes the DescriptionEmbeddingSimilarity metric.

        Args:
            topics (List[str]): The topics to evaluate.

        Returns:
            float: The computed metric.
        """
        if topics is None:
            return 0
        
        result = 0.0
        with torch.no_grad():
            E = self._encoder.encode_text(clip.tokenize([c.lower() for c in topics]).to(device), context_length=477)
        E = E / E.norm(dim=-1, keepdim=True)
        E = E.cpu().numpy()
        # Performing cosine similarity between topic descriptions
        similarity = np.sum(1 - pairwise_distances(E, metric="cosine") - np.diag(np.ones(len(topics))))
        result = similarity / (len(topics) * (len(topics) - 1))
        return result
