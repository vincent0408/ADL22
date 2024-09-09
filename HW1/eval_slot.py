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
from seqeval.metrics import classification_report, accuracy_score
from seqeval.scheme import IOB2

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, slot2idx, args.max_len)
    eval_dl = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                         pin_memory=True, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(embeddings=embeddings,
                    hidden_size=args.hidden_size,
                    dropout=args.dropout,
                    num_class=dataset.num_classes).to(args.device)
    print(model)

    ckpt = torch.load(args.ckpt_path, map_location=torch.device(args.device))
    model.load_state_dict(ckpt)

    predict_lst = []
    true_lst = []
    total_len = []
    eval_correct = 0
    eval_total = 0

    with torch.no_grad():
        for inputs, labels, length in eval_dl:
            inputs, labels, length = inputs.to(args.device), labels.to(args.device), length.to(args.device)
            outputs = model(inputs).permute(0, 2, 1)

            _, predicted = torch.max(outputs.data, 1)
            

            total_len.extend(length)
            for pd, lb, l in zip(predicted, labels, length):
                if(torch.equal(pd[:l], lb[:l])):
                    eval_correct += 1   
                
                pr = []
                la = []                 
                for x, y in zip(pd[:l], lb[:l]):
                    pr.append(dataset.idx2label(x.item()))
                    la.append(dataset.idx2label(y.item()))
                predict_lst.append(pr)
                true_lst.append(la)
            eval_total += labels.size(0)

        print("Joint Acc:", eval_correct/eval_total)
        print("Token Acc:", accuracy_score(predict_lst, true_lst))        
        print(classification_report(predict_lst, true_lst,scheme=IOB2, mode='strict'))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/eval.json"
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
        default="./ckpt/slot/slot-model.pt",
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