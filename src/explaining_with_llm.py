"""
Classes and functions for explaining the clusters using LLMs.
"""

import clip
import torch
import ollama

import argparse
from artwork_clustering import load_model

from typing import List, Tuple
import pickle
import warnings

# Setting some things up
warnings.filterwarnings("ignore", category=FutureWarning)


device = "cuda" if torch.cuda.is_available() else "cpu"



configuration = """
FROM mistral
SYSTEM '''
The user will provide a prompt containing four ordered list of terms describing a cluster of artworks.
The lists are the following: GENRE, TOPIC, MEDIA and STYLE. The terms in each list are ordered from most to least relevant for the cluster.
Provide a concise and organic description which will best explain the cluster using the given terms.
'''
"""

def create_cluster_explainer() -> None:
    """
    Creates the cluster explainer.
    
    Returns:
        None
    """
    ollama.create(
        model="cluster-explainer",
        modelfile=configuration
    )


class Explainer:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = "models/finetuned-v2.pt",
                 groups: List[str] = ["GENRE", "TOPIC", "MEDIA", "STYLE"]) -> None:
        """
        Initializes the explainer.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to "models/finetuned-v2.pt".
            groups (List[str]): The groups to explain. Defaults to ["GENRE", "TOPIC", "MEDIA", "STYLE"].
        """
        create_cluster_explainer()
        self._model = load_model(base_model, model_path)
        self._groups = groups
    

    def __call__(self, interps: List[List[Tuple[str, float]]]) -> None:
        """
        Explains the given interpretations.

        Args:
            interps (List[List[Tuple[str, float]]]): The interpretations to explain.

        Returns:
            None
        """
        self._prompts = [self.construct_prompt(interp) for interp in interps]
        # Generating explanations
        self._explanations = [self.explain(prompt) for prompt in self._prompts]

        # Saving the results
        with open("results/explanations.pkl", "wb") as f:
            pickle.dump({
                "prompts": self._prompts,
                "explanations": self._explanations,
                "similarity": self._explanations_similarity()
            }, f)

    def construct_prompt(self, interp: List[Tuple[str, float]]) -> str:
        """
        Constructs a prompt for the given interpretation.

        Args:
            interp (List[Tuple[str, float]]): The interpretation to construct a prompt for.

        Returns:
            str: The constructed prompt.
        """
        prompt = ""

        for group, terms in zip(self._groups, interp):
            prompt += f"{group}:\n"
            for term, score in terms:
                prompt += f"- {term} (score: {score:.2f})\n"
            prompt += "\n"
        
        return prompt
    
    def explain(self, prompt: str) -> str:
        """
        Explains the given prompt using the cluster explainer.

        Args:
            prompt (str): The prompt to explain.

        Returns:
            str: The explanation.
        """
        response = ollama.chat(
            model="cluster-explainer",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]
    
    def _explanations_similarity(self) -> List[float]:
        """
        Computes the similarity between the explanations.

        Returns:
            List[float]: The similarity between the explanations.
        """
        similarities = []
        explanations = []
        for explanation in self._explanations:
            explanation = clip.tokenize(explanation).to(device)
            explanation = self._model.encode_text(explanation)
            explanation = explanation / explanation.norm(dim=-1, keepdim=True)
            explanations.append(explanation)
        
        # Iterating over the explanations
        for i, explanation in enumerate(explanations):
            total_sim = 0
            # Iterating over the other explanations
            for j, other_explanation in enumerate(explanations):
                if i == j:
                    continue
                # Computing the cosine similarity
                similarity = 100.0 * explanation @ other_explanation.t()
                total_sim += similarity.item()
            similarities.append(total_sim / (len(self._explanations) - 1))
        return similarities





if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model", type=str, default="models/finetuned-v2.pt")
    parser.add_argument("--interps_path", type=str, default="results/kmeans.pkl")

    args = parser.parse_args()

    # 1. load the interpretations
    with open(args.interps_path, "rb") as f:
        interps = pickle.load(f)["interps"]
    
    # 2. initialize the explainer
    explainer = Explainer(
        model_path=args.finetuned_model
    )

    # 3. explain the clusters found
    explainer(interps)
