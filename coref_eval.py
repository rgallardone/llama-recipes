import os
import re
import subprocess
import sys

import fire
import pandas as pd
from transformers import LlamaTokenizer

from configs.datasets import ancora_co_es, ancora_co_es_ud, ancora_co_es_ud_mention
from eval.eval_utils import (
    batch_inference_documents,
    create_conllu_response_file,
    postprocess_coref_result,
    write_dataset_to_conll,
)
from ft_datasets.ancora_dataset import AncoraDataset
from ft_datasets.ancora_dataset_ud import AncoraDatasetUD
from ft_datasets.ancora_dataset_ud_mention import AncoraDatasetUDMention
from ft_datasets.coref_instructions import INSTRUCTIONS, INSTRUCTIONS_MENTION
from inference.model_utils import load_model, load_peft_model

KEY_FILE = "test.key"
RESPONSE_FILE = "test.response"
RESULTS_FILE = "results.parquet"


def main(
    eval_dir: str,
    model_name: str,
    is_UD: bool = True,
    task: str = "cluster_id",
    results_file: str = None,
    peft_model_name: str = None,
    quantization: bool = False,
    max_sentences_per_doc: int = None,
    batch_size: int = 5,
    max_padding_length: int = 1500,
    max_new_tokens=500,  # The maximum numbers of tokens to generate
    use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    **kwargs,
):
    if results_file:
        print("Loading results file for evaluation")
        results_df = pd.read_parquet(results_file)
    else:
        print("Loading models")
        model = load_model(model_name, quantization)
        print("-> Base model loaded")
        if peft_model_name:
            model = load_peft_model(model, peft_model_name)
            print("-> PEFT model loaded")

        print("Loading tokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        model.resize_token_embeddings(model.config.vocab_size + 1)

        print("Load dataset")
        if task == "mention_detection":
            dataset_config = ancora_co_es_ud_mention()
            dev_dataset = AncoraDatasetUDMention(
                dataset_config, tokenizer, "dev"
            ).dataset_split
            inst_first = INSTRUCTIONS_MENTION["inst_first"]
            inst_next = INSTRUCTIONS_MENTION["inst"]
        else:
            if is_UD:
                dataset_config = ancora_co_es_ud()
                dev_dataset = AncoraDatasetUD(
                    dataset_config, tokenizer, "dev"
                ).dataset_split
            else:
                dataset_config = ancora_co_es()
                dev_dataset = AncoraDataset(
                    dataset_config, tokenizer, "dev"
                ).dataset_split
            inst_first = INSTRUCTIONS["next_sentence"]["es"]["inst1_first"]
            inst_next = INSTRUCTIONS["next_sentence"]["es"]["inst1"]

        dev_df = dev_dataset.to_pandas()

        def get_sentence_index(sentence_id):
            match = re.search(r"(\d+)([A-Z]?)$", sentence_id)
            id = int(match.group(1))
            if (match.group(2) != "") and (match.group(2) > "A"):
                id += ord(match.group(2)) - ord("A")
            return id

        dev_df["sentence_index"] = dev_df.groupby("file")["sentence_id"].cumcount() + 1

        if max_sentences_per_doc is not None:
            dev_df = dev_df[dev_df["sentence_index"] <= max_sentences_per_doc]

        print("Run inference over the test data")

        results_df = batch_inference_documents(
            dev_df,
            tokenizer,
            model,
            inst_first,
            inst_next,
            postprocess_coref_result,
            batch_size,
            max_padding_length,
            max_new_tokens,
            use_cache,
            **kwargs,
        )

        print("Store results")
        results_df.to_parquet(f"{eval_dir}/{RESULTS_FILE}")

    if task == "cluster_id":
        if not is_UD:
            print("Create key file")
            if os.path.isfile(f"{eval_dir}/{KEY_FILE}"):
                print("--> Key file already exists")
            else:
                write_dataset_to_conll(
                    results_df, "gold_sentence", f"{eval_dir}/{KEY_FILE}"
                )
                print("--> Key file created")

        print("Create response file")
        if is_UD:
            dev_key_file_path = (
                f"{dataset_config.data_dir}/es_ancora-corefud-dev.conllu"
            )
            create_conllu_response_file(
                results_df, dev_key_file_path, f"{eval_dir}/{RESPONSE_FILE}"
            )
        else:
            write_dataset_to_conll(results_df, "result", f"{eval_dir}/{RESPONSE_FILE}")

        print("Evaluate the results")
        if is_UD:
            corefud_scorer = subprocess.Popen(
                [
                    "python",
                    "corefud-scorer/corefud-scorer.py",
                    dev_key_file_path,
                    f"{eval_dir}/{RESPONSE_FILE}",
                ]
            )
            corefud_scorer.communicate()
        else:
            perl_script = subprocess.Popen(
                [
                    "perl",
                    "reference-coreference-scorers/scorer.pl",
                    "all",
                    f"{eval_dir}/{KEY_FILE}",
                    f"{eval_dir}/{RESPONSE_FILE}",
                    "none",
                ],
                stdout=sys.stdout,
            )
            perl_script.communicate()


if __name__ == "__main__":
    fire.Fire(main)
