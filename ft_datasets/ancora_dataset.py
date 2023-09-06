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

RANDOM_SEED = 42
TRAIN_SIZE = 0.8

instructions = {
    "full_text": {
        "en": {
            "inst1": """Find all the entities that corefer on the following text. You should enclose
             the corefering entities with square brackets and add a number that identifies each
             entity, like this: '[1 John] is a farmer in [2 Arizona]. [1 He] is the best on
             [2 the state]'."""
        },
        "es": {
            "inst1": """Encuentra todas las entidades que correfieren en el siguiente texto.
             Encierra cada entidad con parentesis rectos y añade un número que identifique cada
             entidad, como en el siguiente ejemplo: '[1 Juan] es un granjero en [2 Uruguay].
             [1 Él] es el mejor de [2 el país].""",
            "inst2": """Identifica que entidades correfieren en el siguiente texto. Las entidades
             fueron encerradas con parentesis rectos, y se debe añadir un número que identifique
             cada entidad dentro de los parentesis rectos, como en el siguiente ejemplo: '[1 Juan]
             es un granjero en [2 Uruguay]. [1 Él] es el mejor de [2 el país].""",
            "inst3": """Identifica que menciones correfieren en el texto. Las menciones fueron
             encerradas con parentesis rectos, y se debe añadir un número dentro de los parentesis
             rectos que identifique la entidad a la que refieren las menciones. Notar que hay
             entidades que pueden estar dentro de otras; en ese caso, se debe detectar el
             identificador para cada entidad anidada. Por ejemplo, para el texto de entrada
             '[ Juan ] es un granjero en [ Uruguay ]. [ Él ] es el mejor de [ el país ].',
             la salida deberia ser '[1 Juan] es un granjero en [2 Uruguay]. [1 Él] es el mejor de
             [2 el país].'. El texto está delimitado por los tokens <texto> y </texto>.""",
        },
    },
    "next_sentence": {
        "es": {
            "inst1": (
                "Dado un texto encerrado por los tokens <texto> y </texto> y una oracion "
                "encerrada por los tokens <oracion> y </oracion> con menciones a entidades, "
                "identificar en la oración a que entidad refiere cada mención. En el texto y la "
                "oracion se identifica con parentesis rectos las menciones a entidades. A su vez, "
                "en el texto se identifica con un número todas las menciones que refieren a una "
                "misma entidad. Se debe agregar un identificador a cada mención en la oración que "
                "la asocie a una entidad en el texto. Si no refiere a una entidad presente en el "
                "texto, agrega un nuevo identificador. Responde unicamente con la oracion resultante "
                "y nada mas."
            ),
            "inst1_first": (
                "Dada una oracion encerrada por los tokens <oracion> y </oracion> con "
                "menciones a entidades, identificar con un número a que entidad refiere cada mención. "
                "En la oración se identifica con parentesis rectos las menciones a entidades. "
                "Las menciones a la misma entidad deben llevar el mismo identificador, y todas las "
                "entidades distintas deben llevar distintos identificadores. Responde unicamente con "
                "la oracion resultante y nada mas."
            ),
        }
    },
}


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
            test_file_names = file_names.drop(train_file_names.index)
            train_dataset_df = dataset_df[dataset_df["file"].isin(train_file_names)]
            test_dataset_df = dataset_df[dataset_df["file"].isin(test_file_names)]

            AncoraDataset.train_dataset = Dataset.from_pandas(
                train_dataset_df, split="train"
            )
            AncoraDataset.test_dataset = Dataset.from_pandas(
                test_dataset_df, split="test"
            )

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
            self._load_data(dataset_config, mode, use_gold_mentions)

        if partition == "train":
            self.dataset_split = AncoraDataset.train_dataset
        else:
            self.dataset_split = AncoraDataset.test_dataset
        self.tokenizer = tokenizer
        self.mode = mode
        self.inst = instructions[mode]["es"][instruction]
        self.inst_first = instructions[mode]["es"][f"{instruction}_first"]
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

        if "entity" in root.attrib:
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
