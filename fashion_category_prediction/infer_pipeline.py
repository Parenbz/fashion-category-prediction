import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from fashion_category_prediction.data.datamodule import FashionData
from fashion_category_prediction.models.rnn import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def infer_main(cfg: DictConfig):
    input_data = FashionData(cfg.data.input_path)

    model = RNN(
        cfg.data.input_size,
        cfg.data.hidden_size,
        cfg.data.num_layers,
        cfg.data.num_classes,
    ).to(device)
    model.load_state_dict(torch.load(cfg.model.path))
    model.eval()

    with torch.no_grad():

        outputs = model(input_data)
        _, predicted = torch.max(outputs.data, 1)

    pd.DataFrame(predicted).to_csv(cfg.model.output_file, index=False)
