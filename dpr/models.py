import torch
from transformers import AutoModel
from torch import nn

class Encoder(torch.nn.Module):
    def __init__(self, config, device):
        super(Encoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(config['model']['text_model_path'])
        self.device = torch.device(device)
        self.dropout = nn.Dropout(0.1)

    def forward(self, encode):
        encode_output = self.encoder(**encode.to(self.device))
        encode_output = self.dropout(encode_output.last_hidden_state)
        encode_output = encode_output[:, 0, :]

        return encode_output


class DPR(torch.nn.Module):
    def __init__(self, question_encoder, passage_encoder, device):
          super(DPR, self).__init__()
          self.device = torch.device(device)
          self.question_encoder = question_encoder.to(self.device)
          self.passage_encoder = passage_encoder.to(self.device)


    def forward(self, question_encode, passage_encode):
        question_encoding = self.question_encoder(question_encode)
        passage_encoding = self.passage_encoder(passage_encode)

        return question_encoding, passage_encoding
    
# loss function
def negative_log_loss(question_vectors, passage_vectors, device):

        scores = torch.matmul(question_vectors, torch.transpose(passage_vectors, 0, 1))

        if len(question_vectors.size()) > 1:
            q_num = question_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = torch.nn.functional.log_softmax(scores, dim=1)
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)

        loss = torch.nn.functional.nll_loss(
            softmax_scores,
            labels,
            reduction="mean",
        )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == labels).to(max_idxs.device).sum()

        return loss, correct_predictions_count.item()