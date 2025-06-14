import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_f1_score

from fashion_category_prediction.data.datamodule import FashionDataset
from fashion_category_prediction.models.rnn import RNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_main(cfg: DictConfig):

    writer = SummaryWriter(log_dir="plots/")

    train_dataset = FashionDataset(cfg.data.train_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    model = RNN(
        cfg.data.input_size,
        cfg.data.hidden_size,
        cfg.data.num_layers,
        cfg.data.num_classes,
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Train the model
    for epoch in range(cfg.train.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(
                -1, cfg.data.sequence_length, cfg.data.input_size
            ).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            writer.add_scalar(
                "F1-micro/train, lr = {}, batch size = {}".format(
                    cfg.train.learning_rate, cfg.train.batch_size
                ),
                multiclass_f1_score(
                    input=outputs, target=labels, num_classes=cfg.data.num_classes
                ),
                epoch,
            )
            writer.add_scalar(
                "F1-weighted/train, lr = {}, batch size = {}".format(
                    cfg.train.learning_rate, cfg.train.batch_size
                ),
                multiclass_f1_score(
                    input=outputs,
                    target=labels,
                    num_classes=cfg.data.num_classes,
                    average="weighted",
                ),
                epoch,
            )
            writer.add_scalar(
                "Loss/train, lr = {}, batch size = {}".format(
                    cfg.train.learning_rate, cfg.train.batch_size
                ),
                loss,
                epoch,
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        cfg.train.num_epochs,
                        i + 1,
                        len(train_loader),
                        loss.item(),
                    )
                )

    torch.save(model.state_dict(), cfg.model.path)

    writer.flush()
