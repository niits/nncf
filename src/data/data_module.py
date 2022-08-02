from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collate_fn import CollateFn


class TinyImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        train_transform,
        val_transform,
        dataset_name: str = "frgfm/imagenette'",
        stratify_column: str = "label",
        batch_size: int = 32,
        num_classes: int = 10,
        image_size: str = "320px",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.stratify_column = stratify_column
        self.train_transform = CollateFn(train_transform)
        self.val_transform = CollateFn(val_transform)
        self.num_classes = num_classes
        self.image_size = image_size

    def prepare_data(
        self,
    ):
        dataset = load_dataset(self.dataset_name, self.image_size)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["valid"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_transform,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, collate_fn=self.val_transform
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=self.val_transform
        )
