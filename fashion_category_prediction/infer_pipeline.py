import hydra
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from omegaconf import DictConfig

from fashion_category_prediction.models.rnn import RNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def infer_main(cfg: DictConfig):
    test_dataset = datasets.FashionMNIST(
        root="data/", train=False, transform=transforms.ToTensor()
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=cfg.data.batch_size, shuffle=False
    )

    model = RNN(
        cfg.data.input_size,
        cfg.data.hidden_size,
        cfg.data.num_layers,
        cfg.data.num_classes,
    ).to(device)
    model.load_state_dict(torch.load(cfg.model.path))
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(
                -1, cfg.data.sequence_length, cfg.data.input_size
            ).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Test Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )
