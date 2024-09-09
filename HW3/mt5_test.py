import logging
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
import argparse
import json

logger = get_logger(__name__)

def main():
    args = parse_args()
    accelerator_log_kwargs = {}
    accelerator = Accelerator(**accelerator_log_kwargs)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    accelerator.wait_for_everyone()

    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    config = AutoConfig.from_pretrained(args.ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.ckpt_path,
        from_tf=bool(".ckpt" in args.ckpt_path),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    prefix = ""
    column_names = raw_datasets["test"].column_names
    text_column = args.text_column
    padding = False

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    test_dataset = processed_datasets["test"]
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    prediction = []
    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
        "do_sample":args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            prediction += decoded_preds
    
    with open(args.output_file, mode='w', encoding='utf-8') as f:
        for i, pred in zip(raw_datasets["test"]["id"], prediction):
            json.dump({"title": pred, "id": i},f, ensure_ascii=False)
            f.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="./data/public.json", 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="./predictions.jsonl", 
        help="A json file containing the prediction data."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="./ckpt",
        required=False,
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default='maintext',
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default='title',
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )    
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    main()
