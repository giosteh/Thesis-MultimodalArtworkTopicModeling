"""
Classes and functions for processing data.
"""

import clip
import torch
import torchvision.transforms as T

# import nltk
# nltk.download("wordnet")
from nltk.corpus import wordnet as wn
from typing import Callable, Dict, List
from rdflib import Graph
import random
import re


device = "cuda" if torch.cuda.is_available() else "cpu"



class PromptBuilder:

    def __init__(self, captions_file: str = "data/artwork_captions.txt") -> None:
        """
        Initializes the prompt builder.

        Args:
            captions_file (str): The file containing the captions. Defaults to "data/artwork_captions.txt".
        """
        self._captions_file = captions_file
        self._sparql_query = """
            PREFIX artgraph: <https://www.gennarovessio.com/artgraph-schema#>

            SELECT ?artworkName ?genreName ?tagName ?mediaName ?styleName ?periodName ?artistName

            WHERE {
                ?artwork a artgraph:Artwork .
                ?artwork artgraph:name ?artworkName .
                OPTIONAL {
                    ?artwork artgraph:hasGenre ?genre .
                    ?genre artgraph:name ?genreName .
                }
                OPTIONAL {
                    ?artwork artgraph:about ?tag .
                    ?tag artgraph:name ?tagName .
                }
                OPTIONAL {
                    ?artwork artgraph:madeOf ?media .
                    ?media artgraph:name ?mediaName .
                }
                OPTIONAL {
                    ?artwork artgraph:hasStyle ?style .
                    ?style artgraph:name ?styleName .
                }
                OPTIONAL {
                    ?artwork artgraph:hasPeriod ?period .
                    ?period artgraph:name ?periodName .
                }
                OPTIONAL {
                    ?artwork artgraph:createdBy ?artist .
                    ?artist artgraph:name ?artistName .
                }
            }
        """
    
    def _build_prompt(self, individual: Dict[str, str], tags_list: List[str]) -> str:
        """
        Builds a prompt for the given artwork, using the information from the SPARQL query.
        
        The prompt is built by concatenating the following strings:
        - genre (if present)
        - media (if present)
        - tags (if present)
        - style (if present)
        - artist (if present)
        - period (if present)

        The final prompt is then stripped of any extra spaces and any parentheses containing numbers are removed.

        Args:
            individual (Dict[str, str]): The dictionary containing the information from the SPARQL query.
            tags_list (List[str]): The list of tags associated with the artwork.

        Returns:
            str: The built prompt.
        """
        genre = str(individual["genre"]).strip().lower() if individual["genre"] else None
        media = str(individual["media"]).strip().lower() if individual["media"] else None
        style = str(individual["style"]).strip().lower() if individual["style"] else None
        artist = str(individual["artist"]).strip().lower() if individual["artist"] else None
        period = str(individual["period"]).strip().lower() if individual["period"] else None

        genre_str = re.sub(r"\([^()]*\)", "", genre) if genre else None
        genre_str = genre_str.replace("painting", "") if genre_str else None
        genre_str = f"{genre_str} " if genre_str else ""

        media_str = f"rendered in {media} " if media else ""

        tags_list = [tag for tag in tags_list if not bool(re.search(r"[&#;]", tag))]
        tags_list = [re.sub(r"[a-z]\.([a-z]\.)*", "", tag) for tag in tags_list]
        tags_list = tags_list[:3]
        tags_str = ", ".join([" ".join(tag.split("-")) for tag in tags_list]) if tags_list else None
        tags_str = tags_str.replace(" and", ",") if tags_str else None
        tag_str = f"displaying {tags_str} " if tags_str else ""

        style_str = f"in a {style} manner " if style else ""

        artist_str = " ".join(artist.split("-")) if artist else None
        artist_str = re.sub(r"[-]+", "", artist_str) if artist_str else None
        artist_str = f"made by {artist_str} " if artist_str else ""

        period_str = period.replace("period", "").replace("painting", "").replace("paintings", "") if period else None
        period_str = f"during {period_str} " if period_str else ""

        prompt = f"{genre_str}painting {media_str}{tag_str}{artist_str}{style_str}{period_str}"
        prompt = " ".join(prompt.split())

        return prompt.strip()

    def __call__(self, kg_path: str = "data/artgraph-rdf/artgraph-facts.ttl") -> None:
        """
        Builds the captions file by querying the knowledge graph at the given path,
        and writing each artwork with its corresponding prompt to the file.

        Args:
            kg_path: The path to the knowledge graph file. Defaults to
                "data/artgraph-rdf/artgraph-facts.ttl".

        Returns:
            None
        """
        kg = Graph()
        kg.parse(kg_path, format="turtle")

        results = kg.query(self._sparql_query)
        individuals = {}

        for row in results:
            name = str(row.artworkName).strip()
            tag_name = row.get("tagName", None)

            if name not in individuals:
                individuals[name] = {
                    "genre": row.get("genreName", None),
                    "style": row.get("styleName", None),
                    "media": row.get("mediaName", None),
                    "artist": row.get("artistName", None),
                    "period": row.get("periodName", None),
                    "tags": []
                }
            # Handling tags
            if tag_name:
                tag_name = str(tag_name).strip().lower()
                if tag_name not in individuals[name]["tags"]:
                    individuals[name]["tags"].append(tag_name)
        
        with open(self._captions_file, "w") as f:
            for name, individual in individuals.items():
                tags = individual["tags"]
                prompt = self._build_prompt(individual, tags)
                f.write(f"{name}\t{prompt}\n")



class ImageAugmenter:
    
    def __init__(self) -> None:
        """
        Initializes the image augmenter.
        """
        _, self._clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self._augment = self._random_augmentation()


    def __call__(self, image: torch.Tensor, prob: float = 1.0) -> torch.Tensor:
        """
        Randomly applies the defined image augmentation pipeline to an input image.

        Args:
            image (torch.Tensor): An image tensor
            prob (float): The probability of applying the augmentation pipeline

        Returns:
            torch.Tensor: The augmented image
        """
        image = T.RandomApply([self._augment], p=prob)(image)
        image = self._clip_preprocess(image)

        return image

    def _random_augmentation(self) -> T.Compose:
        """
        Generates a random augmentation pipeline.

        The pipeline is composed of a sequence of randomly chosen transforms,
        which are the following:

        - Random rotation of 30 degrees
        - Random resized crop with a scale of .8 to 1
        - Random horizontal flip
        - Random vertical flip
        - Color jittering with a brightness, contrast and saturation of .2
        - Gaussian noise with a mean of .0 and a standard deviation of .02

        Returns:
            T.Compose: A callable that can be used to transform an image.
        """
        augmentation = T.Compose([
            T.ToTensor(),
            T.RandomChoice([
                T.RandomRotation(degrees=30),
                T.RandomResizedCrop(size=(224, 224), scale=(.8, 1.0)),
                T.RandomHorizontalFlip(p=1.0),
                T.RandomVerticalFlip(p=1.0),
                T.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
                T.Lambda(lambda x: self._gaussian_noise(x))
            ]),
            T.ToPILImage()
        ])

        return augmentation

    def _gaussian_noise(self, image: torch.Tensor, mean: float = .0, std: float = .02) -> torch.Tensor:
        """
        Add Gaussian noise to an image.

        Args:
            image (torch.Tensor): Input image (Tensor)
            mean (float): Mean of the Gaussian distribution
            std (float): Standard deviation of the Gaussian distribution

        Returns:
            torch.Tensor: Noisy image (Tensor)
        """
        noise = torch.randn(image.size()) * std + mean
        return image + noise


class TextAugmenter:
    
    def __init__(self) -> None:
        """
        Initializes the text augmenter.
        """
        self._augment = self._random_augmentation()
    

    def __call__(self, text: str, prob: float = 1.0) -> str:
        """
        Randomly applies the defined text augmentation pipeline to an input text.

        Args:
            text (str): Input text
            prob (float): Probability of applying the augmentation pipeline

        Returns:
            str: Augmented text
        """
        text = self._random_apply(self._augment, p=prob)()(text)

        return text

    def _random_apply(self, func: Callable[[str], str], p: float = .5) -> Callable[[str], str]:
        """
        Applies a given function with a given probability.

        Args:
            func (Callable[[str], str]): The function to be applied
            p (float): The probability of applying the function

        Returns:
            Callable[[str], str]: The chosen function
        """
        if random.uniform(0, 1) > p:
            return lambda x: x
        
        return func

    def _random_augmentation(self) -> Callable[[], Callable[[str], str]]:
        """
        Returns a callable that returns a random text augmentation function.

        The random text augmentation function will be chosen from the following:
        - synonym replacement
        - random swap
        - random insertion
        - random deletion

        Returns:
            Callable[[], Callable[[str], str]]: A callable that can be used to augment text.
        """
        def augmentation() -> Callable[[str], str]:
            augmentations = [
                self._synonym_replacement,
                self._random_swap,
                self._random_insertion,
                self._random_deletion
            ]
            return random.choice(augmentations)

        return augmentation

    def _synonym_replacement(self, text: str, p: float = .2) -> str:
        """
        Replace a random word in the text with a random synonym.

        The probability of replacing a word is given by the parameter p.
        If a word does not have any synonyms, it is left unchanged.

        Args:
            text (str): The input text
            p (float): The probability of replacing a word

        Returns:
            str: The text with a random word replaced with a random synonym
        """
        words = text.split()
        new_words = words.copy()

        for i in range(len(words)):
            if random.uniform(0, 1) > p:
                continue
            synonyms = wn.synsets(words[i])
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                if synonym != words[i]:
                    new_words[i] = synonym
        
        return " ".join(new_words)

    def _random_swap(self, text: str) -> str:
        """
        Swap two random words in the text.

        The two words are chosen randomly from the text.
        The positions of the words are swapped.

        Args:
            text (str): The input text

        Returns:
            str: The text with two random words swapped
        """
        words = text.split()
        new_words = words.copy()
        idx1, idx2 = random.sample(range(len(words)), 2)

        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return " ".join(new_words)
    
    def _random_insertion(self, text: str) -> str:
        """
        Insert a random synonym of a random word in the text.

        The synonym is chosen randomly from the synsets of the random word.
        The position of the insertion is chosen randomly in the text.

        Args:
            text (str): The input text

        Returns:
            str: The text with a random synonym inserted
        """
        words = text.split()
        random_synonym = self._synonym_replacement(random.choice(words), p=1.0)
        random_idx = random.randint(0, len(words) - 1)

        new_words = words[:random_idx] + [random_synonym] + words[random_idx:]
        return " ".join(new_words)
    
    def _random_deletion(self, text: str, p: float = .2) -> str:
        """
        Delete random words in the text with probability p.

        The words are chosen randomly from the text.

        Args:
            text (str): The input text
            p (float): The probability of deleting a word

        Returns:
            str: The text with random words deleted
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        new_words = [w for w in words if random.uniform(0, 1) > p]
        return " ".join(new_words) if new_words else random.choice(words)
