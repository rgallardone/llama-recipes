import math
import re

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import LlamaTokenizer, PreTrainedModel


def convert_to_conll(target_string):
    # Convert a string with coreference clusters marked with square brackets
    # to the CoNLL format
    id_stack = []
    words = []
    corefs = []
    coref_string = ""
    prev_word = ""

    regex = re.compile(r'\[\d+|[a-zA-Z0-9À-ÿ]+\s*|[.,;!?"\'()$%&\\+*-:<=>@_]|\]')
    tokens = re.findall(regex, target_string)

    for t in tokens:
        if re.match(r"\[\d+", t):
            if prev_word != "":
                if coref_string == "":
                    coref_string = "-"
                words.append(prev_word)
                corefs.append(coref_string)
                coref_string = ""
                prev_word = ""

            # Token is an opening bracket with id
            id = re.search(r"\d+", t).group(0)
            id_stack.append(id)
            coref_string += f"({id}"
        elif re.match(r"\]", t):
            # Token is a closing bracket
            id = id_stack.pop()
            if coref_string == "":
                coref_string = f"{id})"
            else:
                open_entity = re.findall(r"\(\d+(?!\))", coref_string)
                if open_entity and open_entity[-1] == f"({id}":
                    coref_string += ")"
                else:
                    coref_string += f"{id})"
        else:
            # Token is a word or punctuation symbol
            if prev_word != "":
                if coref_string == "":
                    coref_string = "-"
                words.append(prev_word)
                corefs.append(coref_string)
                coref_string = ""
                prev_word = ""

            prev_word = t
    if prev_word != "":
        if coref_string == "":
            coref_string = "-"
        words.append(prev_word)
        corefs.append(coref_string)
        coref_string = ""
        prev_word = ""
    return pd.Series(
        {"word_id": list(range(len(words))), "word": words, "coref": corefs}
    )


def convert_dataset_to_conll(df: pd.DataFrame, target_col: str):
    # DEPRECATED
    line = "{file}   {sentence_id}   {word_id}   {word}   -   -   -   -   -   -   -   -   {coref_string}"
    df[["word_id", "word", "coref"]] = df.apply(
        lambda row: convert_to_conll(row[target_col]), axis=1
    )
    df = df.explode(["word_id", "word", "coref"])
    lines_list = df.apply(
        lambda row: line.format_map(
            {
                "file": row.name,
                "sentence_id": row["sentence_id"],
                "word_id": row["word_id"],
                "word": row["word"],
                "coref_string": row["coref"],
            }
        ),
        axis=1,
    )
    return lines_list


def write_dataset_to_conll(df: pd.DataFrame, target_col: str, file_name: str):
    with open(file_name, "w") as f:
        df[["word_id", "word", "coref"]] = df.apply(
            lambda row: convert_to_conll(row[target_col]), axis=1
        )
        df = df.explode(["word_id", "word", "coref"])
        document_groups = df.groupby(["file"])
        for file, file_df in document_groups:
            f.write(f"#begin document ({file});\n")
            sentences_groups = file_df.groupby(["sentence_id"])
            for sent_id, sentence_df in sentences_groups:
                sentence_df = sentence_df.sort_values(by="word_id", ascending=True)
                word_lines = sentence_df.apply(
                    lambda row: f"{sent_id}\t0\t{row['word_id']}\t-\t{row['coref']}\n",
                    axis=1,
                )
                f.writelines(word_lines)
                f.write("\n")
            f.write("#end document\n\n")


def batch_inference_documents(
    df: pd.DataFrame,
    tokenizer: LlamaTokenizer,
    model: PreTrainedModel,
    instructions: dict,
    batch_size: int = 5,
    max_padding_length: int = 2000,
    max_new_tokens=100,  # The maximum numbers of tokens to generate
    use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    **kwargs,
) -> pd.DataFrame:
    # TODO: move this to some general place because AncoraDataset also uses them
    prompt_first = "<s>[INST]{instruction}\n<oracion>{sentence}</oracion>[/INST]"
    prompt = "<s>[INST]{instruction}\n<texto>{previous_text}</texto>\n<oracion>{sentence}</oracion>[/INST]"

    import ipdb

    ipdb.set_trace()
    df = df.set_index("file")
    df["result"] = None
    df["previous_text"] = ""

    for sid in range(df["sentence_id"].max()):
        print(f"Processing sentence_id = {sid}")
        if sid > 0:
            previous_df = results_df[results_df["sentence_id"] == (sid - 1)].copy()
            previous_df.loc[:, "previous_text"] = (
                previous_df["previous_text"] + " " + previous_df["result"]
            )
            sentences_df = (
                df[df["sentence_id"] == sid].copy().drop(columns=["previous_text"])
            )
            sentences_df = sentences_df.join(previous_df["previous_text"])

            prompts = sentences_df.apply(
                lambda row: prompt.format_map(
                    {
                        "instruction": instructions["next_sentence"]["es"]["inst1"],
                        "previous_text": row["previous_text"],
                        "sentence": row["sentence"],
                    }
                ),
                axis=1,
            )
        else:
            sentences_df = df[df["sentence_id"] == sid].copy()
            prompts = sentences_df.apply(
                lambda row: prompt_first.format_map(
                    {
                        "instruction": instructions["next_sentence"]["es"][
                            "inst1_first"
                        ],
                        "sentence": row["sentence"],
                    }
                ),
                axis=1,
            )

        prompts = list(prompts)
        num_batches = math.ceil(len(prompts) / batch_size)

        for i in tqdm(range(num_batches)):
            batch = tokenizer(
                prompts[i * batch_size : (i + 1) * batch_size],
                padding="max_length",
                truncation=True,
                max_length=max_padding_length,
                return_tensors="pt",
            )
            batch = {k: v.to("cuda") for k, v in batch.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    **kwargs,
                )
            outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            sentences_df.iloc[
                i * batch_size : (i + 1) * batch_size,
                sentences_df.columns.get_loc("result"),
            ] = outputs_text

            # Free up GPU memory
            for _, v in batch.items():
                del v
            torch.cuda.empty_cache()

        sentences_df.loc[:, "result"] = sentences_df["result"].apply(
            lambda res: res.split("[/INST]")[-1]
        )

        if sid == 0:
            results_df = sentences_df
        else:
            results_df = pd.concat([results_df, sentences_df])

    return results_df
