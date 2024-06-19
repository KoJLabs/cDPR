import numpy as np
import torch
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup

from models import Encoder, DPR, negative_log_loss
from evaluate import eval_model

import warnings
import wandb
from dpr_dataloader import parse_data
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer
from torch import nn
import os
## 4,8,16,32

warnings.filterwarnings(action='ignore')

wandb.login()
wandb.init()


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


def main(config):
    device = torch.device(config['model']['device'])

    # model load
    question_encoder = Encoder(config, device)
    passage_encoder = Encoder(config, device)

    model = DPR(question_encoder, passage_encoder, device)    

    # Training dataset
    train_data_loader = parse_data(config, 'train')

    # Test dataset
    test_data_loader = parse_data(config, 'valid')

    # optimizer setting
    optimizer = AdamW(model.parameters(), 
                    lr=config['hyper_params']['learning_rate'],
                    weight_decay=config['hyper_params']['weight_decay'],
                    eps=config['hyper_params']['adam_eps'],
                    betas=tuple(config['hyper_params']['adam_betas']),
                    correct_bias=False
                    )
    
    total_steps = len(train_data_loader) * config['hyper_params']['epochs']
    
    scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=config['hyper_params']['warmup_steps'],
                    num_training_steps=total_steps
                )
    
    encoder_tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_path'])
    
    torch.cuda.empty_cache()
    # start training
    for epoch in tqdm(range(1, config['hyper_params']['epochs']+1)):
        
        train_epoch(model, train_data_loader, negative_log_loss, optimizer, scheduler, epoch, device, config, encoder_tokenizer)

        # evaluate model
        eval_loss, eval_accuracy = eval_model(model, test_data_loader, optimizer, negative_log_loss, device, config, encoder_tokenizer)

        wandb.log({"eval_loss": eval_loss, "eval_accuracy": eval_accuracy})
        
        model_name = config['model']['save_model_path'] + f'_{epoch}.pt'

        os.makedirs(config['model']['save_model_path'], exist_ok=True)

        torch.save(model.state_dict(), model_name)
        print("Save model!")

             
    print('Training finished!')
    wandb.finish()
    

# train
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, epoch, device, config, encoder_tokenizer):
    model.train()
    losses = []
    accuracy = []

    for i, batch in tqdm(enumerate(data_loader)):
        question_batch, positive_passage_batch  = batch['question'], batch['positive_passage']

        passage_encoding = encoder_tokenizer(positive_passage_batch, truncation=True, padding=True, max_length=config['hyper_params']['tokenizer_max_length'], return_tensors = 'pt').to(device)
    
        question_encoding = encoder_tokenizer(question_batch, truncation=True, padding=True, max_length=102, return_tensors = 'pt').to(device)

        
        questions_cls, passages_cls = model(question_encoding, passage_encoding)
    
        loss, correct_count = loss_fn(questions_cls, passages_cls, device)

        # log
        accuracy_in_batch = correct_count / config['hyper_params']['batch_size']
        loss_in_batch = loss.item()

        wandb.log({"train_loss_in_batch": loss_in_batch, "train_accuracy": accuracy_in_batch, "epochs":epoch})

        accuracy.append(accuracy_in_batch)
        losses.append(loss_in_batch)

        #backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)