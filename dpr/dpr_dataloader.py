import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from functools import partial
import json

class DPR_Dataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.question = df['questions'].to_numpy()
        self.positive_passage = df['contexts'].to_numpy()
        self.ctx_idx = df['context_id'].to_numpy()
        self.answer = df['answers'].to_numpy()

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, idx):
        return {
            "question": self.question[idx],
            "positive_passage": self.positive_passage[idx],
            "ctx_idx": self.ctx_idx[idx],
            "answer": self.answer[idx]
        }
    

# def collate_fn(batch_size, batch):
#     unique_ctx_idx = list(set([item["ctx_idx"] for item in batch]))  # Get unique ctx_idx values in the batch
#     batch_data = {
#         "question": [],
#         "positive_passage": [],
#         "ctx_idx": unique_ctx_idx,
#         "answer": []
#     }

#     for item in batch:
#         batch_data["question"].append(item["question"])
#         batch_data["positive_passage"].append(item["positive_passage"])
#         batch_data["answer"].append(item["answer"])

#     if len(batch_data["ctx_idx"]) == batch_size:
#         return batch_data
#     else:
#         return None
    

def collate_fn(batch_size, batch):
    # ctx_idx를 키로 사용하여 유니크한 아이템을 저장할 딕셔너리를 초기화합니다.
    unique_ctx_items = {}

    for item in batch:
        # 현재 아이템의 ctx_idx가 유니크한 경우에만 딕셔너리에 추가합니다.
        if item['ctx_idx'] not in unique_ctx_items:
            unique_ctx_items[item['ctx_idx']] = item

    # 유니크한 아이템들로부터 최종 배치 데이터를 구성합니다.
    batch_data = {
        "question": [],
        "positive_passage": [],
        "ctx_idx": [],
        "answer": []
    }

    for item in unique_ctx_items.values():
        batch_data["question"].append(item["question"])
        batch_data["positive_passage"].append(item["positive_passage"])
        batch_data["ctx_idx"].append(item["ctx_idx"])
        batch_data["answer"].append(item["answer"])


    if len(batch_data["ctx_idx"]) == batch_size:
        return batch_data
    else:
        return None

    
    

def parse_data(config, dtype):
    dataset = pd.read_parquet(config['data'][dtype])
    
    if config['data']['inference']:
        dtype = "test"

    if dtype == "train":
        data_loader = DataLoader(
                                    DPR_Dataset(dataset), 
                                    batch_size=config['hyper_params']['batch_size'], 
                                    shuffle=True, 
                                    num_workers=5,
                                    pin_memory=True,
                                    drop_last=True,
                                    persistent_workers=True,
                                    collate_fn=partial(collate_fn, config['hyper_params']['batch_size'])
                            )
    else:
        data_loader = DataLoader(
                                DPR_Dataset(dataset), 
                                batch_size=config['hyper_params']['batch_size'], 
                                shuffle=False, 
                                num_workers=5,
                                pin_memory=True,
                                drop_last=False,
                                persistent_workers=True,
                                collate_fn=partial(collate_fn, config['hyper_params']['batch_size'])
                    )
    
    data_loader = list(filter(lambda x: x is not None, data_loader))        

    return data_loader