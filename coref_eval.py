import os
import subprocess
import sys

import fire
from transformers import LlamaTokenizer

from configs.datasets import ancora_co_es
from eval.eval_utils import batch_inference_documents, write_dataset_to_conll
from ft_datasets.ancora_dataset import AncoraDataset
from ft_datasets.instructions import INSTRUCTIONS
from inference.model_utils import load_model, load_peft_model

KEY_FILE = "eval/data/test.key"
RESPONSE_FILE = "eval/data/test.response"


def main(
    model_name,
    peft_model_name: str = None,
    quantization: bool = False,
    max_sentences_per_doc: int = None,
    batch_size: int = 5,
    max_padding_length: int = 1500,
    max_new_tokens=500,  # The maximum numbers of tokens to generate
    use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    **kwargs,
):
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

    dataset_config = ancora_co_es()

    print("Load dataset")
    test_dataset = AncoraDataset(dataset_config, tokenizer, "test").dataset_split

    test_df = test_dataset.to_pandas()

    if max_sentences_per_doc is not None:
        test_df = test_df[test_df["sentence_id"] <= max_sentences_per_doc]

    import ipdb

    ipdb.set_trace()

    print("Run inference over the test data")

    results_df = batch_inference_documents(
        test_df,
        tokenizer,
        model,
        INSTRUCTIONS,
        batch_size,
        max_padding_length,
        max_new_tokens,
        use_cache,
        **kwargs,
    )

    ipdb.set_trace()

    print("Create key file")
    if os.path.isfile(KEY_FILE):
        print("--> Key file already exists")
    else:
        # TODO: some entities are not closing, check the first document
        write_dataset_to_conll(results_df, "gold_sentence", KEY_FILE)
        print("--> Key file created")

    print("Create response file")
    write_dataset_to_conll(results_df, "result", RESPONSE_FILE)

    print("Evaluate the results")
    perl_script = subprocess.Popen(
        [
            "perl",
            "reference-coreference-scorers/scorer.pl",
            "all",
            KEY_FILE,
            RESPONSE_FILE,
        ],
        stdout=sys.stdout,
    )
    perl_script.communicate()


if __name__ == "__main__":
    fire.Fire(main)
