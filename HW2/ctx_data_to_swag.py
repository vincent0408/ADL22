import pandas as pd
import json
import argparse
import os

os.makedirs('./data/', exist_ok = True)
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
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None, 
    )           
    args = parser.parse_args()
    return args

args = parse_args()
with open( args.context_file,'r', encoding='utf-8') as f:
    ctx = json.load(f)


def train_to_swag():
    train = pd.read_json(args.train_file)
    label_idx = []
    for par, idx in zip(train['paragraphs'].to_list(), train['relevant'].to_list()):
        label_idx.append(par.index(idx))
    label_idx = pd.DataFrame(label_idx, columns = ['labels'])
    choices = []
    for x, y, z, w in train['paragraphs'].to_list():
        choices.append([ctx[x], ctx[y], ctx[z], ctx[w]])
    choices = pd.DataFrame(choices, columns = ['ending0','ending1','ending2','ending3'])

    train_df = pd.concat([train['id'], train['id'], train['question'], train['question'], train['question'], pd.DataFrame(['gold'] * len(label_idx)), choices, label_idx], axis=1)
    train_df.columns = ["video-id" ,"fold-ind" ,"startphrase", "sent1","sent2", "gold-source", "ending0","ending1","ending2","ending3","labels"]

    train_df.to_csv('./data/ctx_train.csv', encoding='utf-8', index=False)

def val_to_swag():
    val = pd.read_json(args.validation_file)
    label_idx = []
    for par, idx in zip(val['paragraphs'].to_list(), val['relevant'].to_list()):
        label_idx.append(par.index(idx))
    label_idx = pd.DataFrame(label_idx, columns = ['labels'])
    choices = []
    for x, y, z, w in val['paragraphs'].to_list():
        choices.append([ctx[x], ctx[y], ctx[z], ctx[w]])
    choices = pd.DataFrame(choices, columns = ['ending0','ending1','ending2','ending3'])

    val_df = pd.concat([val['id'], val['id'], val['question'], val['question'], val['question'], pd.DataFrame(['gold'] * len(label_idx)), choices, label_idx], axis=1)
    val_df.columns = ["video-id" ,"fold-ind" ,"startphrase", "sent1","sent2", "gold-source", "ending0","ending1","ending2","ending3","labels"]
    val_df.to_csv('./data/ctx_val.csv', encoding='utf-8', index=False)
def test_to_swag():
    choices = []
    for x, y, z, w in test['paragraphs'].to_list():
        choices.append([ctx[x], ctx[y], ctx[z], ctx[w]])
    choices = pd.DataFrame(choices, columns = ['ending0','ending1','ending2','ending3'])
    test_df = pd.concat([test['id'], test['id'], test['question'], test['question'], test['question'], pd.DataFrame(['gold'] * len(choices)), choices, pd.DataFrame([0] * len(choices))], axis=1)
    test_df.columns = ["video-id" ,"fold-ind" ,"startphrase", "sent1","sent2", "gold-source", "ending0","ending1","ending2","ending3","labels"]
    test_df.to_csv('./data/ctx_test.csv', encoding='utf-8', index=False)

if args.test_file is not None:
    test = pd.read_json(args.test_file)
    test_to_swag()
else:
    train_to_swag()
    val_to_swag()