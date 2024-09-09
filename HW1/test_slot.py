import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader
from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, slot2idx, args.max_len)

    test_dl = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                         pin_memory=True, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(embeddings=embeddings,
                    hidden_size=args.hidden_size,
                    num_class=dataset.num_classes).to(args.device)
    print(model)

    ckpt = torch.load(args.ckpt_path, map_location=torch.device(args.device))
    model.load_state_dict(ckpt)

    predict_lst = []
    total_len = []
    with torch.no_grad():
        for i, (inputs, length) in enumerate(test_dl):
            inputs, length = inputs.to(args.device), length.to(args.device)
            outputs = model(inputs).permute(0, 2, 1)

            _, predicted = torch.max(outputs.data, 1)
            
            for p in predicted:
                t = []
                for q in p:
                    t.append(dataset.idx2label(q.item()))
                predict_lst.append(t)
            total_len.extend(length)

        with open(args.pred_file, 'w') as f:
            f.write('id,tags\n')
            for (i, predict), l in zip(enumerate(predict_lst), total_len):
                f.write('test-{},{}\n'.format(i, ' '.join(predict[:l])))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/slot-model.pt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)