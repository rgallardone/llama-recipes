import copy
import os
import re
import xml.etree.ElementTree as ET
from typing import Literal, Tuple

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm

from .coref_instructions import INSTRUCTIONS

RANDOM_SEED = 42
POSTPROCESSED_DATASET_DIR = "postprocessed/next_sentence_gold_mentions"


class AncoraDatasetUD(torch.utils.data.Dataset):
    train_dataset = None
    dev_dataset = None

    def _load_data(
        self,
        dataset_config,
        split: str = "train",
    ) -> pd.DataFrame:
        texts = []
        gold_texts = []
        files = []
        previous_texts = []
        sentence_ids = []

        file_path = f"{dataset_config.data_dir}/es_ancora-corefud-{split}.conllu"

        with open(file_path) as f:
            # Read the whole CoNLL-U file
            dataset_string = f.read()

        document_strings = dataset_string.split("# newdoc")

        for document_str in tqdm(document_strings):
            if document_str == "":
                continue

            sentence_strings = document_str.split("# sent_id")

            document_id = re.search(r"id =\s(.*?)\n", sentence_strings[0]).group(1)

            previous_text = ""

            for sent_str in sentence_strings[1:]:
                lines = sent_str.split("\n")

                sentence_id = re.search(r"=\s(.*)", lines[0]).group(1)

                sentence_text = ""
                sentence_gold_text = ""

                line_index = 1

                while line_index < len(lines[1:]) + 1:
                    line = lines[line_index]
                    if line == "" or line[0] == "#":
                        line_index += 1
                        continue
                    attributes = line.split("\t")

                    if "-" in attributes[0]:
                        # Special case of multi-token words
                        attributes_next_line = lines[line_index + 1].split("\t")
                        misc_str_start = attributes_next_line[-1]
                        attributes_next_line = lines[line_index + 2].split("\t")
                        misc_str_end = attributes_next_line[-1]

                    else:
                        misc_str_start = attributes[-1]
                        misc_str_end = misc_str_start

                    if "." in attributes[0]:
                        # It's an empty node
                        line_word = ""
                    else:
                        line_word = attributes[1]

                    if "Entity" in misc_str_start:
                        entity_string = re.search(
                            r"Entity=(.*?)(\||$)", misc_str_start
                        ).group(1)

                        entity_starts = re.findall(r"\(e\d+", entity_string)

                        for e_str in entity_starts:
                            if len(e_str) == 3:
                                entity_id = int(e_str[-1])
                            else:
                                # If the entity number has two many digits, keep only the
                                # last two of them
                                entity_id = int(e_str[-2:])
                            sentence_gold_text += f"[{entity_id} "
                            sentence_text += "[ "

                    sentence_text += line_word + " "
                    sentence_gold_text += line_word + " "

                    if "Entity" in misc_str_end:
                        entity_string = re.search(
                            r"Entity=(.*?)(\||$)", misc_str_end
                        ).group(1)

                        entity_ends = re.findall(r"\)", entity_string)
                        for _ in entity_ends:
                            sentence_gold_text += "] "
                            sentence_text += "] "

                    if "-" in attributes[0]:
                        # Skip split token lines since they should already be processed
                        line_index += 3
                    else:
                        line_index += 1

                sentence_gold_text = re.sub(r" +", " ", sentence_gold_text)
                sentence_text = re.sub(r" +", " ", sentence_text)
                texts.append(sentence_text)
                gold_texts.append(sentence_gold_text)
                sentence_ids.append(sentence_id)
                files.append(document_id)
                previous_texts.append(previous_text)
                previous_text += f" {sentence_gold_text}"

            previous_text = ""

        dataset_df = pd.DataFrame(
            {
                "file": files,
                "sentence_id": sentence_ids,
                "previous_text": previous_texts,
                "sentence": texts,
                "gold_sentence": gold_texts,
            }
        )

        datasets_dir = f"{dataset_config.data_dir}/{POSTPROCESSED_DATASET_DIR}"

        dataset_df.to_parquet(f"{datasets_dir}/{split}.parquet")

        return dataset_df

    def __init__(
        self,
        dataset_config,
        tokenizer,
        partition="train",
        max_words: int = 1500,
    ):
        if AncoraDatasetUD.train_dataset is None:
            datasets_dir = f"{dataset_config.data_dir}/{POSTPROCESSED_DATASET_DIR}"
            if os.path.isdir(datasets_dir) and len(os.listdir(datasets_dir)) > 1:
                train_dataset_df = pd.read_parquet(f"{datasets_dir}/train.parquet")
                dev_dataset_df = pd.read_parquet(f"{datasets_dir}/dev.parquet")
            else:
                train_dataset_df = self._load_data(dataset_config, "train")
                dev_dataset_df = self._load_data(dataset_config, "dev")
            AncoraDatasetUD.train_dataset = Dataset.from_pandas(
                train_dataset_df, split="train"
            )
            AncoraDatasetUD.dev_dataset = Dataset.from_pandas(
                dev_dataset_df, split="dev"
            )

        if partition == "train":
            self.dataset_split = AncoraDatasetUD.train_dataset
        else:
            self.dataset_split = AncoraDatasetUD.dev_dataset
        self.tokenizer = tokenizer
        self.inst = INSTRUCTIONS["next_sentence"]["es"]["inst1"]
        self.inst_first = INSTRUCTIONS["next_sentence"]["es"][f"inst1_first"]
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
