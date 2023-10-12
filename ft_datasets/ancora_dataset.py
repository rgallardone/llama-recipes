import copy
import os
import re
import xml.etree.ElementTree as ET
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm

from .coref_instructions import INSTRUCTIONS

RANDOM_SEED = 42
TRAIN_SIZE = 0.74
DEV_SIZE = 0.12
POSTPROCESSED_DATASET_DIR = "postprocessed/next_sentence_gold_mentions"


class AncoraDataset(torch.utils.data.Dataset):
    train_dataset = None
    test_dataset = None

    def _load_data(
        self,
        dataset_config,
        mode: Literal["full_text", "next_sentence"] = "next_sentence",
        use_gold_mentions: bool = True,
    ):
        texts = []
        gold_texts = []
        files = []
        previous_texts = []
        sentence_ids = []

        file_dirs = ["CESS-CAST-A", "CESS-CAST-AA", "CESS-CAST-P"]
        file_names = []
        for dir in file_dirs:
            for file_name in os.listdir(f"{dataset_config.data_dir}/{dir}"):
                file_names.append(f"{dir}/{file_name}")

        for file_path in tqdm(file_names):
            root = ET.parse(f"{dataset_config.data_dir}/{file_path}").getroot()

            if mode == "full_text":
                text, gold_text = self._parse_tree(root, use_gold_mentions)
                texts.append(re.sub(r" +", " ", text))
                gold_texts.append(re.sub(r" +", " ", gold_text))
                files.append(file_path)
            elif mode == "next_sentence":
                previous_text = ""
                for s in range(len(root)):
                    sentence_text, sentence_gold_text = self._parse_tree(
                        root[s], use_gold_mentions
                    )
                    sentence_gold_text = re.sub(r" +", " ", sentence_gold_text)
                    texts.append(re.sub(r" +", " ", sentence_text))
                    gold_texts.append(sentence_gold_text)
                    previous_texts.append(previous_text)
                    sentence_ids.append(s)
                    files.append(file_path)
                    previous_text += f" {sentence_gold_text}"

        if mode == "full_text":
            # TODO: change splitting stage for full_text mode
            AncoraDataset.dataset = Dataset.from_dict(
                {"text": texts, "gold_text": gold_texts, "file": files}
            )
        elif mode == "next_sentence":
            dataset_df = pd.DataFrame(
                {
                    "file": files,
                    "sentence_id": sentence_ids,
                    "previous_text": previous_texts,
                    "sentence": texts,
                    "gold_sentence": gold_texts,
                }
            )

            file_names = dataset_df["file"].drop_duplicates()
            train_file_names = file_names.sample(
                frac=TRAIN_SIZE, random_state=RANDOM_SEED
            )
            test_dev_file_names = file_names.drop(train_file_names.index)
            dev_frac = DEV_SIZE / (1 - TRAIN_SIZE)
            dev_file_names = test_dev_file_names.sample(
                frac=dev_frac, random_state=RANDOM_SEED
            )
            test_file_names = test_dev_file_names.drop(dev_file_names.index)
            train_dataset_df = dataset_df[dataset_df["file"].isin(train_file_names)]
            dev_dataset_df = dataset_df[dataset_df["file"].isin(dev_file_names)]
            test_dataset_df = dataset_df[dataset_df["file"].isin(test_file_names)]

            datasets_dir = f"{dataset_config.data_dir}/{POSTPROCESSED_DATASET_DIR}"

            train_dataset_df.to_parquet(f"{datasets_dir}/train.parquet")
            dev_dataset_df.to_parquet(f"{datasets_dir}/dev.parquet")
            test_dataset_df.to_parquet(f"{datasets_dir}/test.parquet")

            AncoraDataset.train_dataset = Dataset.from_pandas(
                train_dataset_df, split="train"
            )
            AncoraDataset.dev_dataset = Dataset.from_pandas(dev_dataset_df, split="dev")

    def __init__(
        self,
        dataset_config,
        tokenizer,
        partition="train",
        mode: Literal["full_text", "next_sentence"] = "next_sentence",
        use_gold_mentions: bool = True,
        instruction: str = "inst1",
        max_words: int = 1500,
    ):
        if AncoraDataset.train_dataset is None:
            datasets_dir = f"{dataset_config.data_dir}/{POSTPROCESSED_DATASET_DIR}"
            if os.path.isdir(datasets_dir) and len(os.listdir(datasets_dir)) > 1:
                train_dataset_df = pd.read_parquet(f"{datasets_dir}/train.parquet")
                dev_dataset_df = pd.read_parquet(f"{datasets_dir}/dev.parquet")
                AncoraDataset.train_dataset = Dataset.from_pandas(
                    train_dataset_df, split="train"
                )
                AncoraDataset.dev_dataset = Dataset.from_pandas(
                    dev_dataset_df, split="dev"
                )
            else:
                self._load_data(dataset_config, mode, use_gold_mentions)

        if partition == "train":
            self.dataset_split = AncoraDataset.train_dataset
        else:
            self.dataset_split = AncoraDataset.dev_dataset
        self.tokenizer = tokenizer
        self.mode = mode
        self.inst = INSTRUCTIONS[mode]["es"][instruction]
        self.inst_first = INSTRUCTIONS[mode]["es"][f"{instruction}_first"]
        self.max_words = max_words

    def _parse_tree(
        self, root: ET.Element, use_gold_mentions: bool = True
    ) -> Tuple[str, str]:
        text = ""
        gold_text = ""
        childs_text = ""
        childs_gold_text = ""

        if "wd" in root.attrib:
            childs_text = re.sub(r"_", " ", root.attrib["wd"])
            childs_gold_text = childs_text
        elif "elliptic" not in root.attrib:
            for i in range(len(root)):
                child_text, child_gold_text = self._parse_tree(
                    root[i], use_gold_mentions
                )
                childs_text += f"{child_text} "
                childs_gold_text += f"{child_gold_text} "

        # Singleton mentions are NOT included on the training and evaluation data
        if ("entity" in root.attrib) and ("singleton" not in root.attrib["entity"]):
            ent_num = re.search(r"\d+", root.attrib["entity"]).group(0)
            if use_gold_mentions:
                text += f"[ {childs_text}] "
            else:
                text += f"{childs_text} "
            gold_text += f"[{ent_num} {childs_gold_text}] "
        else:
            text = childs_text
            gold_text = childs_gold_text

        return text, gold_text

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        example = self.dataset_split[index]
        if example["previous_text"] == "":
            prompt = f"<s>[INST]{self.inst_first}\n<oracion>{example['sentence']}</oracion>[/INST]"
        else:
            prompt = f"""<s>[INST]{self.inst}\n<texto>{example['previous_text']}</texto>
            \n<oracion>{example['sentence']}</oracion>[/INST]"""
        prompt_w_output = prompt + f" {example['gold_sentence']} </s>"

        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)

        prompt_w_output = self.tokenizer.encode(prompt_w_output)
        prompt_w_output.append(self.tokenizer.eos_token_id)
        prompt_w_output = torch.tensor(prompt_w_output, dtype=torch.int64)

        padding = self.max_words - prompt_w_output.shape[0]
        if padding > 0:
            prompt_w_output = torch.cat(
                (prompt_w_output, torch.zeros(padding, dtype=torch.int64) - 1)
            )
        else:
            prompt_w_output = prompt_w_output[: self.max_words]

        labels = copy.deepcopy(prompt_w_output)
        labels[: len(prompt)] = -1
        prompt_w_output_mask = prompt_w_output.ge(0)
        labels_mask = labels.ge(0)
        prompt_w_output[~prompt_w_output_mask] = 0
        labels[~labels_mask] = IGNORE_INDEX
        prompt_w_output_mask = prompt_w_output_mask.float()

        return {
            "input_ids": prompt_w_output,
            "labels": labels,
            "attention_mask": prompt_w_output_mask,
        }
