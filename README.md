# Control Token with Dense Passage Retrieval

## MRC Dataset
Dataset from AI Hub: [MRC Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=569)

#### Sample Table (The original dataset is in Korean, but only the samples are translated into English.)
| intention | question                                                         | answer                | context                                                                                         | answer_start | answer_length |
|-----------|------------------------------------------------------------------|-----------------------|-------------------------------------------------------------------------------------------------|--------------|---------------|
| Location  | What is the investment site recommended by Busan as a four-season luxury complex marine leisure city?   | East Busan Tourist Complex       | 'Here is the best place to invest in the coastal development project' - 17 projects including the hinterland development of Jindo Port ...                            | 192          | 8             |
| Etcs      | How many religious facilities are subject to steeple safety management, investigation, and demolition?              | 288 churches          | Steeple Safety Inspection and Demolition Support Plan\nⅠ Overview\n Background\n○ Due to strong winds, typhoons ...                            | 441          | 8             |
| Food  | What is the name of the treatment specially imported and provided to 135 patients for COVID-19 treatment?   | Remdesivir            | COVID-19 domestic occurrence status (regular briefing)\n□ Central Disease Control Headquarters reported that ...                      | 62           | 5             |
| Science  | Where was the 2018 Digital Content Flagship Project launch ceremony held?            | Nuri Dream Square in Sangam-dong  | Expansion of the digital content flagship project through convergence between virtual/augmented content and industry ...                      | 220          | 11            |
| Science  | Which company won the 2015 Korea Safety Technology Award with a smart agent?     | Minister of Public Safety and Security Award  | Ministry of Future Planning, temporary permission for products that were difficult to release due to lack of formal approval - ‘Intelligent Fire Evacuation Guidance ...                       | 472          | 9             |

## Data Preprocessing 
- Context Text Chunking 
    1. Split the text by \n
    2. Further split sentences using the kss package
    3. Recombine the split sentences into 512 token lengths (to match vector length for DPR training)
    4. Identify the index of the context sentence where the answer to the question exists  

## DPR Training
The traditional DPR method uses question vector and context vector for training.

However, our proposed cDPR uses the intention of the question as a special token for training.

Specifically, it trains using intention token + question vector and intention token + context vector.

We use an in-batch-negative size of 32, as larger batch sizes generally improve performance. Negatives are generated using different passages and are ensured not to be duplicated within the same batch.


## Intention Token Prediction Model 
#### Fine-tuning the Intention Token Prediction Model
text: question of MRC dataset, labels: Intention of MRC dataset ➡️ xml-roberta

#### Intention Token Inference Method
question ➡️ Classification model ➡️ Intention token

## Evaluation Method
Evaluate by checking if the correct context is retrieved when a question is posed to the evaluation dataset.
Measure DPR model performance (top 1, 5, 10, 15, 20)

Increase the threshold for intention token inference and evaluate the results.

#### DPR base vs cDPR (ours)

| Metric | DPR base | cDPR (ours) |
|--------|----------|-------------|
| Top1   | 51.1%    | 64.4%       |
| Top5   | 77.8%    | 85.7%       |
| Top10  | 84.9%    | 90.4%       |
| Top15  | 87.8%    | 92.5%       |
| Top20  | 89.6%    | 93.8%       |

#### cDPR with different classification thresholds

| Threshold | Top1 | Top5 | Top10 | Top15 | Top20 |
|-----------|------|------|-------|-------|-------|
| ≥ 0       | 61.7%| 83.6%| 88.3% | 90.9% | 92.3% |
| ≥ 0.5     | 62.5%| 84.2%| 88.8% | 91.4% | 92.6% |
| ≥ 0.7     | 63.4%| 85.0%| 89.6% | 91.9% | 93.3% |
| ≥ 0.9     | 64.4%| 85.7%| 90.4% | 92.5% | 93.8% |

## Getting Started
To run the experiments included in this study, follow these setup instructions.

### Prerequisites
Ensure you have Python 3.10 or newer installed on your system. You may also need to install additional Python libraries, which can be found in the pyproject.toml file:

```
poetry shell
poetry install
```

### Installation
Clone the repository to your local machine:

```
git clone https://github.com/KoJLabs/cDPR
cd cDPR
```


### Running the Preprocess Script
To start the data ordering process, use the following command:

```
python preprocess.py --data_path {data_path} --model_path {drp enocder model_path} --chunk_size {chunk_size}
```

Example
``` 
python preprocess.py --data_path ./datasets/admin_clean_mrc_1.parquet --model_path 'klue/roberta-base' --chunk_size 512
```


### Running the Training Script
To start the training process, use the following command:

```
python dpr/train.py --config_path dpr/configs/train_batch32.yaml
```



## Citation
```
@misc{KoTAN,
  author       = {Juhwan Lee, Jisu Kim},
  title        = {Control Token with Dense Passage Retrieval},
  howpublished = {\url{https://github.com/KoJLabs/cDPR}},
  year         = {2024},
}
```

```
@article{lee2024control,
  title={Control Token with Dense Passage Retrieval},
  author={Lee, Juhwan and Kim, Jisu},
  journal={arXiv preprint arXiv:2405.13008},
  year={2024}
}
```