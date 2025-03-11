"""
Classes and functions for fine-tuning the CLIP model.
"""

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import ImageAugmenter, TextAugmenter
from typing import Tuple
from PIL import Image
import argparse
import pickle
import torch
import clip
import os


# Setting the seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"



def load_model(base_model: str, model_path: str, return_preprocess: bool = False) -> nn.Module:
    """Loads a pretrained or fine-tuned CLIP model.

    Args:
        base_model (str): The base model to use.
        model_path (str): The path to the finetuned model.
        return_preprocess (bool, optional): Whether to return the preprocess. Defaults to False.

    Returns:
        nn.Module: The finetuned model.
    """
    model, preprocess = clip.load(base_model, device=device, jit=False)    
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    model.float()
    model.eval()

    if return_preprocess:
        return model, preprocess
    return model


class ImageCaptionDataset(Dataset):

    def __init__(self,
                 images_dir: str = "data/images/imagesf2",
                 captions_file_path: str = "data/artwork_captions.txt",
                 apply_augmentations: bool = False,
                 path_only: bool = False):
        """Initializes the ImageTextDataset.

        Args:
            images_dir (str): The directory containing the images. Defaults to "images/imagesf2".
            captions_file (str): The file containing the captions. Defaults to "artwork_captions.txt".
            apply_augmentations (bool): Whether to apply augmentations. Defaults to False.
            path_only (bool): Whether to only return the image path and text. Defaults to False.
        """
        self.apply_augmentations = apply_augmentations
        self._path_only = path_only
        self._images_dir = images_dir

        _, self._clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self._image_augmenter = ImageAugmenter()
        self._text_augmenter = TextAugmenter()

        with open(captions_file_path, "r") as f:
            lines = f.readlines()
        self._image_caption_pairs = [(line.split("\t")[0], line.split("\t")[1].strip()) for line in lines]
    

    def __len__(self) -> int:
        """Returns the number of image-caption pairs in the dataset.

        Returns:
            int: The number of image-caption pairs.
        """
        return len(self._image_caption_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Returns the image and text at a given index.

        Args:
            idx (int): The index of the image and text to be returned.

        Returns:
            Tuple[torch.Tensor, str]: A tuple containing the image and text at the given index.
        """
        image_path, text = self._image_caption_pairs[idx]
        image_path = os.path.join(self._images_dir, image_path)
        if self._path_only:
            return image_path, text
        image = Image.open(image_path).convert("RGB")
        # Apply augmentations
        if self.apply_augmentations:
            image = self._image_augmenter(image)
            text = self._text_augmenter(text)
        else:
            image = self._clip_preprocess(image)

        return image, text


class CLIPFinetuner:

    def __init__(self,
                 model_name: str = "ViT-B/32",
                 dataset: Dataset = ImageCaptionDataset(),
                 val_split: float = .3,
                 batch_size: int = 128,
                 lr: float = 5e-5,
                 augment: bool = False,
                 unfreeze_from: int = 6,
                 unfreeze_every: int = 2,
                 models_dir: str = "models"):
        """Initializes the CLIPFinetuner.

        Args:
            model_name (str, optional): The name of the model. Defaults to "ViT-B/32".
            dataset (Dataset, optional): The dataset to use. Defaults to ImageCaptionDataset().
            val_split (float, optional): The proportion of the dataset to use for validation. Defaults to .3.
            batch_size (int, optional): The batch size. Defaults to 128.
            lr (float, optional): The learning rate. Defaults to 5e-5.
            augment (bool, optional): Whether to apply augmentations. Defaults to False.

            unfreeze_from (int, optional): The number of blocks to unfreeze. Defaults to 6.
            unfreeze_every (int, optional): The number of blocks to unfreeze every time. Defaults to 2.
            models_dir (str, optional): The directory to save the models. Defaults to "models".
        """
        self._model, _ = clip.load(model_name, device=device, jit=False)
        self._model.float()
        self._dataset = dataset

        self._tot_blocks = len(self._model.visual.transformer.resblocks)
        self._unfreezing_completed = False
        self._freeze_model()
        self._unfreeze_blocks(1)
        self._unfreeze_from = unfreeze_from
        self._unfreeze_every = unfreeze_every
        # Training settings
        self._early_stopping = EarlyStopping(self._model)
        self._train_loader, self._val_loader = self._get_dataloaders(val_split, batch_size, augment)
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=.2)

        self._resume_epoch = 0
        self._models_dir = models_dir
    

    def load_model(self, path: str) -> None:
        """Loads the model from a checkpoint.

        Args:
            path (str): The path to the checkpoint.

        Returns:
            None
        """
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._resume_epoch = checkpoint["epoch"]

    def _save_model(self, epoch: int) -> None:
        """Saves the model checkpoint.

        Args:
            epoch (int): The current epoch.

        Returns:
            None
        """
        last_checkpoint_path = os.path.join(self._models_dir, f"checkpoint-e{epoch}.pt")
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
        checkpoint_path = os.path.join(self._models_dir, f"checkpoint-e{epoch+1}.pt")
        
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "epoch": epoch+1
        }, checkpoint_path)
    
    def _get_dataloaders(self, val_split: float = .3, batch_size: int = 128, apply_augmentations: bool = False) -> Tuple[DataLoader, DataLoader]:
        """Creates the train and validation DataLoaders.

        Args:
            val_split (float, optional): The proportion of the dataset to include in the validation set. Defaults to .3.
            batch_size (int, optional): The batch size for the DataLoaders. Defaults to 128.
            apply_augmentations (bool, optional): Whether to apply data augmentation. Defaults to False.

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing the train and validation DataLoaders.
        """
        train_size = int((1 - val_split) * len(self._dataset))
        val_size = len(self._dataset) - train_size
        train_dataset, val_dataset = random_split(self._dataset, [train_size, val_size])

        train_dataset.dataset.apply_augmentations = apply_augmentations
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    
    def fit(self, epochs: int = 100, verbose: bool = True) -> None:
        """Trains the model for the given number of epochs.

        Args:
            epochs (int, optional): The number of epochs to train the model. Defaults to 100.
            verbose (bool, optional): Whether to print the training and validation losses and scores during training. Defaults to True.

        Returns:
            None
        """
        tot_epochs = epochs + self._resume_epoch
        for epoch in range(self._resume_epoch, tot_epochs):
            self._unfreeze_model(epoch)
            train_loss = self._train()
            val_loss, val_score = self._validate()

            self._save_model(epoch)
            if verbose:
                print(f"\nEpoch #{epoch+1}/{tot_epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")
            # Checking for early stopping
            stop = self._early_stopping(train_loss, val_loss, val_score)
            if stop:
                if verbose:
                    print(f"Early stopping at epoch #{epoch+1}!")
                break
    
    def _freeze_model(self) -> None:
        """Freeze all the model's parameters.
        """
        for p in self._model.parameters():
            p.requires_grad_(False)

    def _unfreeze_blocks(self, blocks_to_unfreeze: int) -> None:
        """Unfreeze the last given number of transformer blocks in the model.

        Args:
            blocks_to_unfreeze (int): The number of blocks to unfreeze.

        Returns:
            None
        """
        if blocks_to_unfreeze == self._tot_blocks:
            for p in self._model.parameters():
                p.requires_grad_()
            self._unfreezing_completed = True
            return
        self._model.visual.proj.requires_grad_()
        self._model.text_projection.requires_grad_()

        for i in range(self._tot_blocks - blocks_to_unfreeze, self._tot_blocks):
            for p in self._model.visual.transformer.resblocks[i].parameters():
                p.requires_grad_()

        for i in range(self._tot_blocks - blocks_to_unfreeze, self._tot_blocks):
            for p in self._model.transformer.resblocks[i].parameters():
                p.requires_grad_()

    def _unfreeze_model(self, epoch: int) -> None:
        """Partially unfreeze the model, given the epoch.

        Every self._unfreeze_every epochs, starting from self._unfreeze_from, unfreeze one more transformer block.

        Args:
            epoch (int): The current epoch.

        Returns:
            None
        """
        if self._unfreezing_completed:
            return
        epoch += 1

        if epoch >= self._unfreeze_from and epoch % self._unfreeze_every == 0:
            blocks_to_unfreeze = (epoch - self._unfreeze_from + self._unfreeze_every) // self._unfreeze_every
            blocks_to_unfreeze += 1

            if blocks_to_unfreeze <= self._tot_blocks:
                self._unfreeze_blocks(blocks_to_unfreeze)
                if self._unfreezing_completed:
                    print(f"\n<All {self._tot_blocks} transformer blocks have been unfrozen!>")
                else:
                    print(f"\n<Unfrozen blocks {self._tot_blocks - blocks_to_unfreeze} to {self._tot_blocks}.>")

    def _clip_score(self, images: torch.Tensor, texts: torch.Tensor) -> float:
        """Computes the CLIP score for the given images and texts.

        Args:
            images (torch.Tensor): The images.
            texts (torch.Tensor): The texts.

        Returns:
            float: The CLIP score.
        """
        with torch.no_grad():
            image_features = self._model.encode_image(images)
            text_features = self._model.encode_text(texts)
        # Normalize and compute score
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.t()).diag().mean()

        return score

    def _train(self) -> float:
        """Train the model on the training set for one epoch.

        Returns:
            float: The average training loss.
        """
        self._model.train()
        total_loss = .0

        for images, texts in self._train_loader:
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)

            logits_per_image, logits_per_text = self._model(images, texts)
            ground_truth = torch.arange(len(images), dtype=torch.long).to(device)
            loss_img = F.cross_entropy(logits_per_image, ground_truth)
            loss_txt = F.cross_entropy(logits_per_text, ground_truth)
            loss = (loss_img + loss_txt) / 2

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

        total_loss /= len(self._train_loader)
        return total_loss

    def _validate(self) -> Tuple[float, float]:
        """Validate the model on the validation set.

        Returns:
            Tuple[float, float]: The average validation loss and score.
        """
        self._model.eval()
        total_loss = .0
        total_score = .0

        with torch.no_grad():
            for images, texts in self._val_loader:
                images = images.to(device)
                texts = clip.tokenize(texts).to(device)

                logits_per_image, logits_per_text = self._model(images, texts)
                ground_truth = torch.arange(len(images), dtype=torch.long).to(device)
                loss_img = F.cross_entropy(logits_per_image, ground_truth)
                loss_txt = F.cross_entropy(logits_per_text, ground_truth)
                loss = (loss_img + loss_txt) / 2

                score = self._clip_score(images, texts)
                total_score += score.item()
                total_loss += loss.item()

        total_score /= len(self._val_loader)
        total_loss /= len(self._val_loader)
        return total_loss, total_score



class EarlyStopping:

    def __init__(self,
                 model: nn.Module,
                 patience: int = 80,
                 models_dir: str = "models",
                 mode: str = "max"):
        """Initialize the early stopping object.

        Args:
            model (nn.Module): The model to be trained.
            patience (int, optional): The number of epochs to wait for improvement. Defaults to 50.
            models_dir (str, optional): The directory path to save the model. Defaults to "models".
            mode (str, optional): The mode of the early stopping. Defaults to "max".
        """
        self._model = model
        self._patience = patience
        self._best_score = None
        self._counter = 0
        self._mode = mode
        self._stop = False
        self._models_dir = models_dir

        self._train_loss = []
        self._val_loss = []
        self._val_scores = []

    def __call__(self, train_loss: float, val_loss: float, val_score: float) -> bool:
        """Check if the early stopping criteria is met.

        Args:
            train_loss (float): The training loss.
            val_loss (float): The validation loss.
            val_score (float): The validation score.

        Returns:
            bool: True if the early stopping criteria is met, False otherwise.
        """
        self._train_loss.append(train_loss)
        self._val_loss.append(val_loss)
        self._val_scores.append(val_score)
        self._save_lists()

        score = val_score
        
        if self._best_score is None:
            self._best_score = score
        elif self._is_improvement(score):
            self._save_checkpoint()
            self._best_score = score
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._stop = True
        
        return self._stop

    def _is_improvement(self, score: float) -> bool:
        """Check if the score is an improvement.

        Args:
            score (float): The score to check.

        Returns:
            bool: True if the score is an improvement, False otherwise.
        """
        if self._mode == "max":
            return score > self._best_score
        else:
            return score < self._best_score

    def _save_checkpoint(self, name: str = "checkpoint.pt") -> None:
        """Saves the model checkpoint.

        Args:
            name (str, optional): The name of the checkpoint. Defaults to "checkpoint.pt".

        Returns:
            None
        """
        checkpoint_path = os.path.join(self._models_dir, name)
        torch.save(self._model.state_dict(), checkpoint_path)
    
    def _save_lists(self) -> None:
        """Saves the lists of losses and scores.
        
        Args:
            None

        Returns:
            None
        """
        data = (self._train_loss, self._val_loss, self._val_scores)
        with open(os.path.join(self._models_dir, "tracking.pkl"), "wb") as f:
            pickle.dump(data, f)



if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=200)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    # 1. initialize the finetuner
    finetuner = CLIPFinetuner(
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        augment=args.augment
    )
    if args.load:
        finetuner.load_model(args.load)

    finetuner.fit(epochs=args.epochs, verbose=True)
