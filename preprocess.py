import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import kss
from multiprocessing import Pool
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


# 병렬 처리할 함수 정의
def process_data(text):
    # 문자열을 줄 단위로 분할하고 kss로 문장 분리
    sentences = text.split('\n')
    split_sentences = kss.split_sentences(sentences, num_workers=1, strip=False, backend='punct')
    return split_sentences

# 데이터를 처리하는 병렬 함수
def parallel_process(df, func, n_cores=30):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# 적용할 함수를 데이터프레임의 컬럼에 적용하는 래퍼 함수
def apply_parallel(df_column):
    return df_column.apply(process_data)

#
def find_answer_in_range_nearby(sentences, answer, start, end):
    current_pos = 0
    matched_indexes = []
    
    for index, sentence in enumerate(sentences):
        sentence_start = current_pos
        sentence_end = current_pos + len(sentence)

        # 해당 범위가 문장 범위 내에 완전히 포함되는지 확인
        if ((sentence_start-len(sentences)) <= start) and ((sentence_end+len(sentences)) >= end):
            answer = answer.strip()
            if answer in sentence:
                matched_indexes.append(index) 
                return matched_indexes
        current_pos += len(sentence)

    return matched_indexes



def main(args):
    # data 불러오기
    df = pd.read_parquet(args.data_path)
    df = df.reset_index(drop=True)

    # model 불러오기
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 전체 데이터프레임에 병렬 처리 적용
    df['split_sentences'] = parallel_process(df['context'], apply_parallel)
    df = df.reset_index(drop=True)

    # set params
    chunk_size = int(args.chunk_size)
    number_of_wrong_split = 0
    max_context = []
    valid_idx = []

    for idx in tqdm(range(len(df))):
        answer = df.iloc[idx]['answer']
        start_idx = df.iloc[idx]['answer_start']
        end_idx = df.iloc[idx]['answer_start'] + df.iloc[idx]['answers_len']

        sentence_parts = df.iloc[idx]['split_sentences']
        flatten_sentence_parts = [ x for xs in sentence_parts for x in xs ]
        
        # 정답값 문장 인덱스 찾기 (중복은 없는지 체크해보자)
        indexes = find_answer_in_range_nearby(flatten_sentence_parts, answer,  start_idx,  end_idx)
        
        if indexes:
                
            # init
            if len(indexes) != 1:
                print('index multiple!')
                
            sentence_idx = indexes[0]
            token_length = len(tokenizer.tokenize(flatten_sentence_parts[sentence_idx]))
            loop_count = 0
            sentence_bundles = []
            substracted_context = None
            start_idx = 0
            end_idx = 0
            flatten_sentences_length = len(flatten_sentence_parts)
            
            # 
            while token_length <= chunk_size and not (start_idx == 0 and end_idx ==flatten_sentences_length) and (end_idx <= flatten_sentences_length):
                if loop_count == 0:
                    substracted_context = flatten_sentence_parts[sentence_idx]
                else:
                    # 
                    if sentence_idx - loop_count <= 0:
                        start_idx = 0
                    else:
                        start_idx = sentence_idx - loop_count
                    # 
                    if sentence_idx + loop_count == flatten_sentences_length:
                        end_idx = flatten_sentences_length
                    else:
                        end_idx = sentence_idx + loop_count
                        
                    # 
                    if start_idx == 0 and end_idx == 0:
                        substracted_context = flatten_sentence_parts[sentence_idx]
                    else:
                        substracted_context = flatten_sentence_parts[start_idx : end_idx]
                        substracted_context = "\n".join(substracted_context)
                                    
                token_length = len(tokenizer.tokenize(substracted_context))
                loop_count +=1
                
                if token_length <= chunk_size:
                    sentence_bundles.append(substracted_context)
                    
            max_context.append(sentence_bundles)                    
            valid_idx.append(idx)
        else:
            print('filtered wrong splited context')
            number_of_wrong_split +=1
        
    print('Total filtered data:', number_of_wrong_split)
    filtered_df = df.iloc[valid_idx]
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['context_bundle'] = max_context
    filtered_df['preprocessed_context'] = [list_[-1:] for list_ in max_context]
    filtered_df['preprocessed_context'] = filtered_df['preprocessed_context'].apply(lambda x: str(x[0]) if x and x[0] is not None else '')
        
    # Splitting the data
    train_df, temp_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Saving the splits to parquet files
    train_df.to_parquet('./dpr_train.parquet')
    val_df.to_parquet('./dpr_val.parquet')
    test_df.to_parquet('./dpr_test.parquet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= './data.parquet')
    parser.add_argument('--model_path', type=str, default='klue/roberta-base')
    parser.add_argument('--chunk_size', type=str, default='512')

    args = parser.parse_args()
    main(args)