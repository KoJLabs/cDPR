import numpy as np
import torch
import argparse

from models import Encoder, DPR

import warnings
import wandb
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer
import faiss
import warnings
import json
import gc
import pandas as pd
from torch.utils.data import DataLoader

gc.collect()

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/db.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    return config


def faiss_index(config, data_loader, encoder_tokenizer):
    index = faiss.IndexFlatIP(1024)
    index = faiss.IndexIDMap2(index)

    device = torch.device(config['model']['device'])
    model = torch.load(config['model']['model_path'], map_location=torch.device('cpu'))

    from collections import OrderedDict

    ordered_dict_passage = OrderedDict()

    for key, value in model.items():
        if key.split('.')[0] == "passage_encoder":
            key = '.'.join(key.split('.')[1:])
            ordered_dict_passage[key] = value

    passage_encoder = Encoder(config['model']['text_model_path'], device)

    passage_encoder.load_state_dict(ordered_dict_passage)
    passage_encoder.to(device)
    passage_encoder.eval()

    with torch.no_grad():
        cnt = 0
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            passage_encoding = encoder_tokenizer(batch, truncation=True, padding=True, max_length=config['hyper_params']['tokenizer_max_length'], return_tensors = 'pt').to(device)

            passage_cls = passage_encoder(passage_encoding)
            
            index.add_with_ids(passage_cls.cpu().numpy().astype('float32'), np.array(range(cnt, cnt + passage_cls.shape[0]))) # (배치 벡터, 배치단위로 추가할 고유 id)

            cnt+=passage_cls.shape[0]


    faiss.write_index(index, config['faiss']['fname'])
    print(index.ntotal)
    print("n total ===============")


def main(config):
    df = pd.read_parquet(config['data']['context'])
    dataset = df['context_512'].tolist()
    data_loader = DataLoader(dataset, batch_size=config['hyper_params']['batch_size'], shuffle=False)
    encoder_tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_path'])
    faiss_index(config, data_loader, encoder_tokenizer)



if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)