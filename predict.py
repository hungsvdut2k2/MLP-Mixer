import os
import torch
from argparse import ArgumentParser
from models.base_model import ImageClassifier
from PIL import Image

if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.getcwd()
    parser.add_argument(
        "--weight-path",
        default="{}/logs/cat_dog_classfication/version_0/checkpoints/epoch=9-step=2040.ckpt".format(
            home_dir
        ),
        type=str,
    )
    parser.add_argument(
        "--image-path",
        default="{}/dataset/small_dog_cat_dataset/test/cats/cat.48.jpg".format(
            home_dir
        ),
        type=str,
    )

    args = parser.parse_args()
    model = ImageClassifier.load_from_checkpoint(args.weight_path).to("cpu")
    image_tensor = torch.tensor(Image.open(args.image_path)).unsqueeze(0)
    print(torch.argmax(model(image_tensor)))
