# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class ancora_co_es:
    dataset: str = "ancora_co_es"
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"
    data_dir: str = "ancora_dataset/ancora-3.0.1es"


@dataclass
class ancora_co_es_ud:
    dataset: str = "ancora_co_es_ud"
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"
    data_dir: str = "corefud_1_1/CorefUD-1.1-public/data/CorefUD_Spanish-AnCora"
