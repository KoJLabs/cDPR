import numpy as np
import torch
import argparse
import pandas as pd

from models import Encoder, DPR

import warnings
from dpr_dataloader import parse_data
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer
import faiss
import warnings
import json
import gc
import os

gc.collect()

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    return config


def faiss_index(config):
    index = faiss.read_index(config['faiss']['path'])
    print(index.ntotal)
    print("n total ===============")

    return index


def retrieval_performance(data_length, labels):
    count_dict = {i+1:0 for i in range(len(labels[0]))}
    for i in range(len(labels)):
        if 1 in labels[i]:
            idx = labels[i].index(1)
            count_dict[idx+1] += 1

    acc_count = 0
    acc_dict = {}

    for k, v in zip(count_dict.keys(), count_dict.values()):
        acc_count += v
        acc_dict[k] = acc_count/data_length
        
    return acc_dict


def inference(config, data_loader, encoder_tokenizer, index, context_list):
    device = torch.device(config['model']['device'])

    model = torch.load(config['model']['model_path'], map_location=torch.device('cpu'))

    from collections import OrderedDict

    ordered_dict_passage = OrderedDict()

    for key, value in model.items():
        if key.split('.')[0] == "question_encoder":
            key = '.'.join(key.split('.')[1:])
            ordered_dict_passage[key] = value

    selection_model_dataset = []

    question_encoder = Encoder(config['model']['text_model_path'], device)
    question_encoder.load_state_dict(ordered_dict_passage)
    question_encoder.to(device)
    question_encoder.eval()
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            question_batch, answer = batch['question'], batch['answer']
            question_encoding = encoder_tokenizer(question_batch, truncation=True, padding=True, max_length=config['hyper_params']['tokenizer_max_length'], return_tensors = 'pt').to(device)

            question_cls = question_encoder(question_encoding)

            ## index 추출
            _, indices = index.search(question_cls.cpu().numpy().astype("float32"), 20)

            for key, value in enumerate(zip(indices, answer)):
                labels = []
                for ctx in context_list[value[0]]:
                    if value[1].replace(' ','') in ctx.replace(' ',''):
                        labels.append(1)
                    else:
                        labels.append(0)
                        
                selection_model_dataset.append(
                    {
                        "question": question_batch[key],
                        "answer": answer[key],
                        "passages": context_list[value[0]].tolist(),
                        "labels": labels
                    }
                )      

        return selection_model_dataset


def main(config):
    dtype = config["data"]["dtype"]   
    
    data_loader = parse_data(config, dtype)
    encoder_tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_path'])
    index = faiss_index(config)

    context_data = pd.read_parquet(config["faiss"]["context"])

    context_list = np.array(context_data["context_512"])
    selection_model_dataset = inference(config, data_loader, encoder_tokenizer, index, context_list)


    if dtype == "test":
        test_labels = []
        for i in range(len(selection_model_dataset)):
            test_labels.append(selection_model_dataset[i]['labels'])
            
        test_acc = retrieval_performance(len(test_labels), test_labels)
        for i in [1, 5, 10, 15, 20]:
            print("Test Accuracy ===============")
            print(i, ':', test_acc[i])


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)