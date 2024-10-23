"""
Main file for finetuning CLIP.
"""


import argparse
from clip_finetuning import CLIPFinetuner



if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=50)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load", type=str, default=None)

    args = parser.parse_args()

    # 1. initialize the finetuner
    finetuner = CLIPFinetuner(
        model_name="ViT-B/32",
        batch_size=args.batch_size,
        lr=args.learning_rate,
        augment=args.augment
    )

    if args.load:
        finetuner.load_model(args.load)

    # 2. finetune the model
    finetuner.fit(epochs=args.epochs, verbose=True)
