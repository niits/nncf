from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collate_fn import CollateFn


class TinyImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        train_transform,
        val_transform,
        dataset_name: str = "Maysee/tiny-imagenet",
        stratify_column: str = "label",
        batch_size: int = 32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.stratify_column = stratify_column
        self.train_transform = CollateFn(train_transform)
        self.val_transform = CollateFn(val_transform)

        self.tiny_imagenet = load_dataset(self.dataset_name, split="train+valid")
        self.num_classes = max(self.tiny_imagenet["label"]) + 1

    def prepare_data(
        self,
    ):

        ds = self.tiny_imagenet.train_test_split(
            test_size=0.2, stratify_by_column=self.stratify_column
        )
        tmp, self.test_dataset = ds["train"], ds["test"]
        ds = tmp.train_test_split(
            test_size=0.2, stratify_by_column=self.stratify_column
        )
        self.train_dataset, self.val_dataset = ds["train"], ds["test"]

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
