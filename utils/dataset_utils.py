# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial
from typing import Optional

import torch

from ft_datasets import (
    get_alpaca_dataset,
    get_ancora_dataset,
    get_ancora_ud_dataset,
    get_ancorda_ud_mention_dataset,
    get_grammar_dataset,
    get_samsum_dataset,
)

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "ancora_co_es": get_ancora_dataset,
    "ancora_co_es_ud": get_ancora_ud_dataset,
    "ancora_co_es_ud_mention": get_ancorda_ud_mention_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
