import pandas as pd
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="./data/train.json", 
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default="./data/valid.json", 
    )
    parser.add_argument(
        "--context_file", 
        type=str, 
        default="./data/context.json", 
    )     
    args = parser.parse_args()
    return args

args = parse_args()
with open( args.context_file,'r', encoding='utf-8') as f:
    ctx = json.load(f)
train = pd.read_json(args.train_file)
val = pd.read_json(args.validation_file)

def train_to_squad():
    for t in train:
        t['title'] = t['id']
        t['context'] = ctx[t['relevant']]
        t['answers'] = {'text':[t['answer']['text']], 'answer_start':[t['answer']['start']]}
        del(t['paragraphs'])
        del(t['answer'])
        del(t['relevant'])
    with open('./data/qa_train.json', 'w', encoding='utf8') as f:
        for t in train:
            json.dump(t, f, ensure_ascii=False)
            f.write('\n')

def val_to_squad():
    for t in val:
        t['title'] = t['id']
        t['context'] = ctx[t['relevant']]
        t['answers'] = {'text':[t['answer']['text']], 'answer_start':[t['answer']['start']]}
        del(t['paragraphs'])
        del(t['answer'])
        del(t['relevant'])
    with open('./data/qa_val.json', 'w', encoding='utf8') as f:
        for v in val:
            json.dump(v, f, ensure_ascii=False)
            f.write('\n')

train_to_squad()
val_to_squad()