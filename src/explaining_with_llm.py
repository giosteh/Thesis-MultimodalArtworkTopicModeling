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
<s>
[INST]
The user will provide four ordered lists of terms describing a cluster of artworks: GENRE, TOPIC, MEDIA, and STYLE.
Each list is ordered from most to least relevant for the cluster. Your task is to generate a single, short description that merges the most relevant terms from all four lists into a concise and cohesive sentence.
Do not add any additional details, context, or narrative beyond the given terms. The output must be a single description that integrates all the lists organically.

For instance:
GENRE: vanitas, allegorical, still life,
TOPIC: flowers, fruits, vegetables,
MEDIA: tempera, oil, pastel,
STYLE: classical, symbolism, abstract,

Output:
A vanitas still life painting blending allegorical symbolism, depicting flowers, fruits, and vegetables in oil and tempera, combining classical and symbolism styles.
[/INST]
</s>
[INST]
From now on, your output must be a single, short description that merges the most relevant terms from all four lists into a concise and cohesive sentence.
Do not add any additional details, context, or narrative beyond the given terms. The output must be a single description that integrates all the lists organically.
[/INST]
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

        for group_name, group in zip(self._groups, interp):
            terms = [term for term, _ in group]
            prompt += f"{group_name}: {', '.join(terms)}\n"
        
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
        with torch.no_grad():
            explanations = clip.tokenize(self._explanations).to(device)
            explanations = self._model.encode_text(explanations)
            explanations = explanations / explanations.norm(dim=-1, keepdim=True)
            
            # Computing the cosine similarity
            similarity = (100.0 * explanations @ explanations.t()).fill_diagonal_(0.0)
            similarities = similarity.sum(dim=-1) / (len(self._explanations) - 1)

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
