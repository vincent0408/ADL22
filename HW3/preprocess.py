import jsonlines
import json
import sys
import os

os.makedirs('./data/', exist_ok=True)

def prepare(data_dir, filename):
    with jsonlines.open(data_dir, 'r') as t:
        lst = [obj for obj in t]

    with open(filename, 'w') as f:
        for jl in lst:
            json.dump({'maintext':jl['maintext'],'id':jl['id']}, f, ensure_ascii=False)
            f.write('\n')

prepare(sys.argv[1], './data/test.json')
