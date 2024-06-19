import numpy as np
import torch


# eval
def eval_model(model, data_loader, optimizer, loss_fn, device, config, encoder_tokenizer):
    losses = []
    accuracy = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            question_batch, positive_passage_batch  = batch['question'], batch['positive_passage']

            passage_encoding = encoder_tokenizer(positive_passage_batch, truncation=True, padding=True, max_length=config['hyper_params']['tokenizer_max_length'], return_tensors = 'pt').to(device)

            question_encoding = encoder_tokenizer(question_batch, truncation=True, padding=True, max_length=102, return_tensors = 'pt').to(device)
            
            questions_cls, passages_cls = model(question_encoding, passage_encoding)    
                
            optimizer.zero_grad()

            loss, correct_count = loss_fn(questions_cls, passages_cls, device)
            losses.append(loss.item())
            accuracy.append(correct_count/config['hyper_params']['batch_size'])

    return np.mean(losses), np.mean(accuracy)