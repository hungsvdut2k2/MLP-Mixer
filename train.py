import pytorch_lightning as pl
import os
from models.base_model import ImageClassifier
from data.build_dataset import build_dataloader
from models.mlp_mixer import MlpMixer
from pytorch_lightning.loggers import CSVLogger
from argparse import ArgumentParser


def train(
    train_data_dir: str,
    valid_data_dir: str,
    batch_size: int,
    num_classes: int,
    patch_size: int,
    s: int,
    c: int,
    ds: int,
    dc: int,
    num_mlp_blocks: int,
    epochs: int,
):
    print("----------------Build Dataset------------------------")
    train_dataloader = build_dataloader(train_data_dir, batch_size)
    valid_dataloader = build_dataloader(valid_data_dir, batch_size)
    print("----------------Finish Build Dataset-----------------")
    model = MlpMixer(patch_size, s, c, ds, dc, num_mlp_blocks, epochs)
    base_model = ImageClassifier(model, num_classes)
    logger = CSVLogger("logs", name="cat_dog_classfication")
    trainer = pl.Trainer(max_epochs=epochs, logger=logger)
    trainer.fit(base_model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = os.getcwd()
    parser.add_argument(
        "--train-folder",
        default="{}/dataset/small_dog_cat_dataset/train".format(home_dir),
        type=str,
    )
    parser.add_argument(
        "--valid-folder",
        default="{}/dataset/small_dog_cat_dataset/test".format(home_dir),
        type=str,
    )
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--dc", default=512, type=int, help="Token-mixing units")
    parser.add_argument("--ds", default=2048, type=int, help="Channel-mixing units")
    parser.add_argument("--s", default=196, type=int)
    parser.add_argument("--c", default=768, type=int, help="Projection units")
    parser.add_argument("--patch-size", default=16, type=int)
    parser.add_argument("--num-mlp-blocks", default=8, type=int)

    args = parser.parse_args()

    train(
        args.train_folder,
        args.valid_folder,
        args.batch_size,
        args.num_classes,
        args.patch_size,
        args.s,
        args.c,
        args.ds,
        args.dc,
        args.num_mlp_blocks,
        args.epochs,
    )
