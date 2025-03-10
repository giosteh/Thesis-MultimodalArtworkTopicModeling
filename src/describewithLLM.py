"""
Classes and functions for describing topics with LLMs.
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from metrics import CaptionEmbeddingSimilarity
from finetuneCLIP import load_model
from typing import List, Tuple
from PIL import Image
import numpy as np
import warnings
import pickle
import clip
import torch

# General settings
warnings.filterwarnings("ignore", category=FutureWarning)


device = "cuda" if torch.cuda.is_available() else "cpu"



BASIC_PROMPT = """
Given this image containing a sample of artworks from a cluster, generate a single sentence overall description of the cluster which must be straight to the point.
Avoid general information and focus only on the most relevant aspects of the artworks.
"""

RICH_PROMPT = """
Given this image containing a sample of artworks from a cluster and the following lists of terms which describe it, generate a single sentence overall description of the cluster which must be straight to the point.
Avoid general information and focus only on the most relevant aspects of the artworks.\n
"""


class Descriptor:

    def __init__(self,
                 embedding_model: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt"),
                 pov_names: List[str] = ["Genre", "Subject", "Medium", "Style"]):
        """Initializes the explainer.

        Args:
            embedding_model (Tuple[str, str]): The embedding model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
            pov_names (List[str]): The pov names. Defaults to ["Genre", "Subject", "Medium", "Style"].
        """
        self._embedding_model = load_model(embedding_model[0], embedding_model[1])
        self._pov_names = pov_names
        self._topics = []

        # Setting up LLM & processor
        self._llm = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self._llm.config.pad_token_id = self._llm.config.eos_token_id
        
        self._processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            vision_feature_select_strategy="default",
            patch_size=14
        )


    def __call__(self,
                 output_dir: str,
                 image_paths: List[str],
                 topics: List[List[str]] = None) -> None:
        """Explains the topics using the LLM.

        Args:
            output_dir (str): The path to the output directory.
            image_paths (List[str]): The paths to the topic images.
            topics (List[List[str]]): The topics to explain.
        
        Returns:
            None
        """
        if topics is None:
            prompts = [BASIC_PROMPT] * len(image_paths)
        else:
            prompts = [self._setup_prompt(topic) for topic in topics]
        # Describing the topics
        descriptions = [
            self.describe(image_path, prompt_text)
            for image_path, prompt_text in zip(image_paths, prompts)
        ]

        metric = CaptionEmbeddingSimilarity()
        score = metric(descriptions)
        print(f"CES: {score:.4f}")
        saving = {"descriptions": descriptions, "score": score}
        # Saving the results
        with open(f"{output_dir}/descr.pkl", "wb") as f:
            pickle.dump(saving, f)
        
        return descriptions, score

    def _setup_prompt(self, topic: List[str] = None) -> str:
        """Sets up the prompt text.

        Args:
            topic (List[str]): The topic to explain.

        Returns:
            str: The prompt text.
        """
        if topic is not None:
            prompt_text = RICH_PROMPT
            povs = np.array_split(topic, len(self._pov_names))
            extra_content = "\n".join(
                f"{pov_name.upper()} : " + ", ".join(povs[i])
                for i, pov_name in enumerate(self._pov_names)
            )
            prompt_text += extra_content
        return prompt_text

    def describe(self, image_path: str, prompt_text: str) -> str:
        """Describe the topic given a sample image using the LLM.

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
        # Preparing the prompt
        prompt = self._processor.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True
        )
        input_ids = self._processor(image, prompt, return_tensors="pt").to(device)
        # Actual generation
        with torch.no_grad():
            output_ids = self._llm.generate(**input_ids, max_new_tokens=100)
        description = self._processor.decode(output_ids[0], skip_special_tokens=True)

        description = str(description).split("[/INST]")[-1].strip()
        print(f"Description: {description}")
        return description
