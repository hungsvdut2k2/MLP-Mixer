import pytorch_lightning as pl
from models.base_model import ImageClassifier
from data.build_dataset import build_dataloader
from models.mlp_mixer import MlpMixer
from pytorch_lightning.loggers import CSVLogger

train_data_dir = "./dataset/small_dog_cat_dataset/train"
valid_data_dir = "./dataset/small_dog_cat_dataset/test"


def train(
    patch_size=16,
    s=196,
    c=768,
    ds=2048,
    dc=512,
    num_mlp_blocks=8,
    num_classes=2,
    epochs=1,
):
    train_dataloader = build_dataloader(train_data_dir, 16)
    valid_dataloader = build_dataloader(valid_data_dir, 16)

    model = MlpMixer(16, 196, 768, 2048, 512, 8, 2)
    base_model = ImageClassifier(model, 2)
    logger = CSVLogger("logs", name="cat_dog_classfication")
    trainer = pl.Trainer(max_epochs=epochs, logger=logger)
    trainer.fit(base_model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    train()
