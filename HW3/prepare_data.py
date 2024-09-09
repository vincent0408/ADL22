import jsonlines
import json


TRAIN = './data/train.jsonl'
PUBLIC = './data/public.jsonl'

def prepare(data_dir, filename):
    with jsonlines.open(data_dir, 'r') as t:
        lst = [obj for obj in t]

    with open(filename, 'w') as f:
        for jl in lst:
            json.dump({'title':jl['title'], 'maintext':jl['maintext'], 'id':jl['id']}, f, ensure_ascii=False)
            f.write('\n')

prepare(TRAIN, './data/train.json')
prepare(PUBLIC, './data/public.json')