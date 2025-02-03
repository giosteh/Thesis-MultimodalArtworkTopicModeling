"""
Classes and functions for explaining the clusters using LLMs.
"""

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from CLIPFinetuning import load_model
from typing import List, Tuple
from PIL import Image
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


class Explainer:

    def __init__(self,
                 embedding_model: Tuple[str, str] = ("ViT-B/32", "models/finetuned-v2.pt"),
                 theme_names: List[str] = ["Genre", "Subject", "Medium", "Style"]) -> None:
        """
        Initializes the explainer.

        Args:
            embedding_model (Tuple[str, str]): The embedding model to use. Defaults to ("ViT-B/32", "models/finetuned-v2.pt").
            theme_names (List[str]): The theme names. Defaults to ["Genre", "Subject", "Medium", "Style"].
        """
        self._embedding_model = load_model(embedding_model[0], embedding_model[1])
        self._theme_names = theme_names
        self._topics = []

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
                 sample_paths: List[str],
                 saving_path: str,
                 topics: List[List[List[Tuple[str, float]]]],
                 rich_prompt: bool = False) -> None:
        """
        Explains the given interpretations.

        Args:
            sample_paths (List[str]): The paths to the sample images.
            topics (List[List[List[Tuple[str, float]]]]): The topics to explain.
            rich_prompt (bool): Whether to use a comprehensive prompt_text.

        Returns:
            None
        """
        nr_topics = len(topics[0])
        self._topics = [
            [t[i] for t in topics] for i in range(nr_topics)
        ]

        prompts = [self._setup_prompt(topic, rich_prompt) for topic in self._topics]
        # Generating the descriptions
        descriptions = [
            self.describe(path, prompt) for path, prompt in zip(sample_paths, prompts)
        ]

        # Saving the descriptions
        saving = {
            "descriptions": descriptions,
            "similarities": self.cross_similarity(descriptions)
        }
        with open(saving_path, "wb") as f:
            pickle.dump(saving, f)

    def _setup_prompt(self, topic: List[List[Tuple[str, float]]], rich_prompt: bool) -> str:
        """
        Sets up the prompt text for explaining the given interpretation.

        Args:
            topic (List[List[Tuple[str, float]]]): The interpretation to explain.
            rich_prompt (bool): Whether to use a comprehensive prompt_text.

        Returns:
            str: The prompt_text.
        """
        prompt_text = BASIC_PROMPT

        if rich_prompt:
            prompt_text = RICH_PROMPT
            for theme, terms in zip(self._theme_names, topic):
                words = [t for t, _ in terms[:3]]
                prompt_text += f"{theme}: {', '.join(words)}\n"
        return prompt_text

    def describe(self, image_path: str, prompt_text: str) -> str:
        """
        Describe the topic given a sample image.

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

        # template = (
        #     "{% for message in messages %}"
        #     "{% if message['role'] != 'system' %}"
        #     "{{ message['role'].upper() + ': '}}"
        #     "{% endif %}"
        #     "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
        #     "{{ '<image>\n' }}"
        #     "{% endfor %}"
        #     "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
        #     "{{ content['text'] + ' '}}"
        #     "{% endfor %}"
        #     "{% endfor %}"
        # )
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

    def cross_similarity(self, descriptions: List[str]) -> List[float]:
        """
        Computes the cross similarity between the descriptions.

        Args:
            descriptions (List[str]): The descriptions.

        Returns:
            List[float]: The cross similarity.
        """
        with torch.no_grad():
            texts = clip.tokenize(descriptions).to(device)
            texts = self._embedding_model.encode_text(texts)
            texts = texts / texts.norm(dim=-1, keepdim=True)

            similarities = (texts @ texts.t()).fill_diagonal_(0).cpu().numpy()
        return (similarities.sum(axis=1) / (similarities.shape[1] - 1)).tolist()
