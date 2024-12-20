"""
Classes and functions for explaining the clusters using LLMs.
"""

import clip
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

import argparse
from artwork_clustering import load_model

from typing import List, Tuple
from PIL import Image
import pickle
import glob
import warnings

# Setting some things up
warnings.filterwarnings("ignore", category=FutureWarning)


device = "cuda" if torch.cuda.is_available() else "cpu"



BASIC_PROMPT = """
Given this image containing a sample of artworks from a cluster, generate a few sentences description of the cluster.
The description should be short and straight to the point, avoiding general information and focusing on the most relevant aspects of the artworks. 
"""


class Explainer:

    def __init__(self,
                 base_model: str = "ViT-B/32",
                 model_path: str = "models/finetuned-v2.pt",
                 groups: List[str] = ["GENRE", "TOPIC", "COLOR", "MEDIA", "STYLE"]) -> None:
        """
        Initializes the explainer.

        Args:
            base_model (str): The base model to use. Defaults to "ViT-B/32".
            model_path (str): The path to the finetuned model. Defaults to "models/finetuned-v2.pt".
            groups (List[str]): The groups to explain. Defaults to ["GENRE", "TOPIC", "COLOR", "MEDIA", "STYLE"].
        """
        self._clip_model = load_model(base_model, model_path)
        self._groups = groups
        self._image_paths = []
        self._prompts = []
        self._descriptions = []

        self._llm = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-13b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self._processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
        self._processor.patch_size = 14
        self._processor.vision_feature_select_strategy = "default"
    
    
    def __call__(self, image_paths: List[str], interps: List[List[Tuple[str, float]]], comprehensive: bool = False) -> None:
        """
        Explains the given interpretations.

        Args:
            image_paths (List[str]): The paths to the cluster sample images.
            interps (List[List[Tuple[str, float]]]): The interpretations to explain.
            comprehensive (bool): Whether to use a comprehensive prompt_text.

        Returns:
            None
        """
        self._image_paths = image_paths
        self._prompts = [self.setup_prompt(interp, comprehensive) for interp in interps]
        # Generating descriptions
        self._descriptions = [self.describe(path, text) for path, text in zip(self._image_paths, self._prompts)]

        # Saving the results
        with open("results/descriptions.pkl", "wb") as f:
            pickle.dump({
                "descriptions": self._descriptions,
                "similarity": self._descriptions_similarity()
            }, f)

    def setup_prompt(self, interp: List[Tuple[str, float]], comprehensive: bool = False) -> str:
        """
        Sets up the prompt text for explaining the given interpretation.

        Args:
            interp (List[Tuple[str, float]]): The interpretation to construct a prompt_text for.
            comprehensive (bool): Whether to use a comprehensive prompt_text.

        Returns:
            str: The prompt_text.
        """
        prompt_text = BASIC_PROMPT
        if not comprehensive:
            return prompt_text

        prompt_text += """
            In the description you'll generate, consider using the following ordered lists of terms.\n\n
        """
        for group_name, group in zip(self._groups, interp):
            terms = [term for term, _ in group]
            prompt_text += f"{group_name}: {', '.join(terms)}\n"
        return prompt_text
    
    def describe(self, image_path: str, prompt_text: str) -> str:
        """
        Describe the cluster using the sample image and the given prompt_text.

        Args:
            image_path (str): The path to the sample image.
            prompt_text (str): The prompt text to use.

        Returns:
            str: The description.
        """
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]

        template = (
            "{% for message in messages %}"
            "{% if message['role'] != 'system' %}"
            "{{ message['role'].upper() + ': '}}"
            "{% endif %}"
            "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
            "{{ '<image>\n' }}"
            "{% endfor %}"
            "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
            "{{ content['text'] + ' '}}"
            "{% endfor %}"
            "{% endfor %}"
        )
        # Infer the description
        prompt = self._processor.apply_chat_template(
            conversation=conversation,
            chat_template=template,
            add_generation_prompt=True
        )
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    
    def _descriptions_similarity(self) -> List[float]:
        """
        Computes the similarity between the descriptions.

        Returns:
            List[float]: The similarity between the descriptions.
        """
        with torch.no_grad():
            descriptions = clip.tokenize(self._descriptions).to(device)
            descriptions = self._clip_model.encode_text(descriptions)
            descriptions = descriptions / descriptions.norm(dim=-1, keepdim=True)
            
            # Computing the cosine similarity
            similarity = (descriptions @ descriptions.t()).fill_diagonal_(.0)
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
    with open(f"{args.target}.pkl", "rb") as f:
        interps = pickle.load(f)["interps"]
    # 1.2 get the sample image paths
    image_paths = sorted(glob.glob(f"{args.target}_cluster*.png"))
    
    # 2. initialize the explainer
    explainer = Explainer(
        model_path=args.finetuned_model
    )

    # 3. describe the clusters
    explainer(image_paths, interps, comprehensive=args.comprehensive)
