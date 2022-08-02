import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision.models.resnet import resnet152


class ResNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        resnet_version,
        optimizer="adam",
        transfer=True,
        lr: float = 1e-3,
    ):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        optimizers = {"adam": Adam, "sgd": SGD}
        self.optimizer = optimizers[optimizer]

        self.criterion = (
            nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        )

        self.resnet_model = resnets[resnet_version](pretrained=transfer)

        linear_size = list(self.resnet_model.children())[-1].in_features

        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):

        imgs, labels = batch["image"], batch["label"]
        predictions = self.resnet_model(imgs)

        loss = self.criterion(predictions, labels)
        _, preds = torch.max(predictions, 1)

        acc = (preds == labels.data).type(torch.FloatTensor).mean()

        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch["image"], batch["label"]
        predictions = self.resnet_model(imgs)
        loss = self.criterion(predictions, labels)

        _, preds = torch.max(predictions, 1)

        acc = (preds == labels.data).type(torch.FloatTensor).mean()

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return acc
        

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = torch.stack(validation_step_outputs)
        self.log("val_acc", all_preds.mean(), logger=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        predictions = self.resnet_model(imgs).argmax(dim=1)
        acc = (labels == torch.argmax(predictions, 1)).float().mean()

        self.log("test_acc", acc)

    def create_model(self, pretrained: bool = True):
        return resnet152(pretrained=pretrained)
