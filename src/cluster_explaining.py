"""
Classes and functions for explaining the clusters using LLMs.
"""

import clip
import torch
import ollama

import argparse
from artwork_clustering import load_model

from typing import List, Tuple
from PIL import Image
from io import BytesIO
import pickle
import glob
import warnings

# Setting some things up
warnings.filterwarnings("ignore", category=FutureWarning)


device = "cuda" if torch.cuda.is_available() else "cpu"



BASIC_PROMPT = """
Given this image containing a sample of artworks from a cluster, provide a concise and organic description of the cluster.
"""


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
        ollama.pull("llava:13b-v1.6")
        self._model = load_model(base_model, model_path)
        self._groups = groups
    

    def __call__(self, image_paths: List[str], interps: List[List[Tuple[str, float]]], comprehensive: bool = False) -> None:
        """
        Explains the given interpretations.

        Args:
            image_paths (List[str]): The paths to the cluster sample images.
            interps (List[List[Tuple[str, float]]]): The interpretations to explain.
            comprehensive (bool): Whether to use a comprehensive prompt.

        Returns:
            None
        """
        self._image_paths = image_paths
        self._prompts = [self.setup_prompt(interp, comprehensive) for interp in interps]
        # Generating descriptions
        self._descriptions = [self.describe(path, prompt) for path, prompt in zip(self._image_paths, self._prompts)]

        # Saving the results
        with open("results/descriptions.pkl", "wb") as f:
            pickle.dump({
                "descriptions": self._descriptions,
                "similarity": self._descriptions_similarity()
            }, f)

    def setup_prompt(self, interp: List[Tuple[str, float]], comprehensive: bool = False) -> str:
        """
        Sets up a prompt for explaining the given interpretation.

        Args:
            interp (List[Tuple[str, float]]): The interpretation to construct a prompt for.
            comprehensive (bool): Whether to use a comprehensive prompt.

        Returns:
            str: The constructed prompt.
        """
        prompt = BASIC_PROMPT
        if not comprehensive:
            return prompt

        prompt += """
            In the description, consider using the following ordered lists of terms resulting from a previous interpretation.\n\n
        """
        for group_name, group in zip(self._groups, interp):
            terms = [term for term, _ in group]
            prompt += f"{group_name}: {', '.join(terms)}\n"
        return prompt
    
    def describe(self, image_path: str, prompt: str) -> str:
        """
        Describe the cluster using the sample image and the given prompt.

        Args:
            image_path (str): The path to the sample image.
            prompt (str): The prompt to use.

        Returns:
            str: The description.
        """
        image = Image.open(image_path).convert("RGB")
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        
        description = ""
        # Generating the description
        for response in ollama.generate(model="llava:13b-v1.6",
                                        prompt=prompt,
                                        images=[image_bytes],
                                        stream=True):
            print(response["response"], end="", flush=True)
            description += response["response"]
        
        return description
    
    def _descriptions_similarity(self) -> List[float]:
        """
        Computes the similarity between the descriptions.

        Returns:
            List[float]: The similarity between the descriptions.
        """
        with torch.no_grad():
            descriptions = clip.tokenize(self._descriptions).to(device)
            descriptions = self._model.encode_text(descriptions)
            descriptions = descriptions / descriptions.norm(dim=-1, keepdim=True)
            
            # Computing the cosine similarity
            similarity = (100.0 * descriptions @ descriptions.t()).fill_diagonal_(0.0)
            similarities = similarity.sum(dim=-1) / (len(self._descriptions) - 1)

        return similarities



if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model", type=str, default="models/finetuned-v2.pt")
    parser.add_argument("--target", type=str, default="results/kmeans05")
    parser.add_argument("--comprehensive", action="store_true")

    args = parser.parse_args()

    # 1.1 load the interpretations
    with open(f"{args.results_name}.pkl", "rb") as f:
        interps = pickle.load(f)["interps"]
    # 1.2 get the sample image paths
    image_paths = sorted(glob.glob(f"{args.results_name}*.png"))
    
    # 2. initialize the explainer
    explainer = Explainer(
        model_path=args.finetuned_model
    )

    # 3. describe the clusters
    explainer(image_paths, interps, comprehensive=args.comprehensive)
