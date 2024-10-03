
from finetuning import CLIPFinetuner


if __name__ == "__main__":
    # finetuning the CLIP model
    finetuner = CLIPFinetuner()
    finetuner.fit(epochs=300, verbose=True)