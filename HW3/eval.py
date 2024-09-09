import json
import argparse
import os

from ckiptagger import WS, data_utils
from rouge import Rouge

try:
    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.join(os.getenv("HOME"), ".cache"))
except:
    cache_dir = 'C:/Users/Vincent/.cache'
download_dir = os.path.join(cache_dir, "ckiptagger")
data_dir = os.path.join(cache_dir, "ckiptagger/data")
os.makedirs(download_dir, exist_ok=True)
if not os.path.exists(os.path.join(data_dir, "model_ws")):
    data_utils.download_data_gdown(download_dir)

ws = WS(data_dir)

def tokenize_and_join(sentences):
    return [" ".join(toks) for toks in ws(sentences)]

rouge = Rouge()

def get_rouge(preds, refs, avg=True, ignore_empty=False):
    """wrapper around: from rouge import Rouge
    Args:
        preds: string or list of strings
        refs: string or list of strings
        avg: bool, return the average metrics if set to True
        ignore_empty: bool, ignore empty pairs if set to True
    """
    if not isinstance(preds, list):
        preds = [preds]
    if not isinstance(refs, list):
        refs = [refs]
    preds, refs = tokenize_and_join(preds), tokenize_and_join(refs)
    return rouge.get_scores(preds, refs, avg=avg, ignore_empty=ignore_empty)

def main(args):
    refs, preds = {}, {}

    with open(args.reference, encoding='utf-8') as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission, encoding='utf-8') as file:
        for line in file:
            line = json.loads(line)
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]
    result = get_rouge(preds, refs)
    result = {'rouge-1':result['rouge-1']['f']*100, 'rouge-2':result['rouge-2']['f']*100, 'rouge-l':result['rouge-l']['f']*100}
    print('-----------------')
    print(result)
    print('Pass:', result['rouge-1'] > 22.0 and result['rouge-2'] > 8.5 and result['rouge-l'] > 20.5)
    print('-----------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', default='./data/public.jsonl')
    parser.add_argument('-s', '--submission', default='./predictions.jsonl')
    args = parser.parse_args()
    main(args)
