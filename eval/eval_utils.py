import math
import re

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import LlamaTokenizer, PreTrainedModel
import difflib


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
            if prev_word == "":
                # If it's a wordless entity, add it on a line by itself
                coref_string += ")"
                words.append("")
                corefs.append(coref_string)
                coref_string = ""
            else:
                if coref_string == "":
                    coref_string = f"{id})"
                else:
                    open_entity = re.findall(r"\(\d+(?![\)\d])", coref_string)
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

    df = df.set_index("file")
    df["raw_result"] = None
    df["result"] = None
    df["previous_text"] = ""

    for sid in range(df["sentence_id"].max()+1):
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

        # Decrease batch size as documents get bigger
        if ((sid % 5 == 0) and (sid > 0)):
            batch_size = batch_size - 1

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
                sentences_df.columns.get_loc("raw_result"),
            ] = outputs_text

            # Free up GPU memory
            for _, v in batch.items():
                del v
            torch.cuda.empty_cache()

        sentences_df.loc[:, "result"] = sentences_df[["sentence", "raw_result"]].apply(
            lambda row: postprocess_result(row["sentence"], row["raw_result"]), axis=1
        )

        if sid == 0:
            results_df = sentences_df
        else:
            results_df = pd.concat([results_df, sentences_df])

    return results_df

def align_result_with_sentence(sentence: str, result: str) -> str:
    """Due to hallucination, the resulting output of the model might not be exactly
    the same as the sentence given to it. This function modifies the result so that
    it's text is exactly the same as the sentence, but with the added identifiers
    to the mention.

    Args:
        sentence (str): target sentence without the identifiers for the mentions.
        result (str): result of the model, with added identifiers for the mentions
        and slight differences with respect to the sentence.

    Returns:
        str: result of the model, with the exact same text as the sentence, and
        added identifiers to the mentions.
    """
    # Remove all preceeding whitespaces, dots and commas on the result
    result = re.sub(r'^[\s.,]*', '', result)
    output = ""
    for _, s in enumerate(difflib.ndiff(result, sentence)):
        if (s[0] == "-" and not re.match(r"\d", s[-1])):
            continue
        output += s[-1]
    return output

def fill_empty_identifiers(result: str) -> str:
    """Model might not output an identifier for a mention. In that case,
    we add a default identifier (999) for the scoring algorithms to work.

    Args:
        result (str): result of the model, with mentions that might not have
        identifiers.

    Returns:
        str: result of the model, where mentions with no identifiers were
        assigned the default identifier 999.
    """
    return re.sub(r"\[(?!\d)", "[999", result)

def fill_identifiers_in_sentence(sentence: str, result: str) -> str:
    id_list = re.findall(r"(?<=\[)\d*", result)

    target_mentions_num = len(re.findall(r"\[", sentence))

    if (len(id_list) < target_mentions_num):
        id_list = ["999" for _ in range(target_mentions_num - len(id_list))] + id_list
    
    id_iter = iter(id_list)

    def get_id_string(match):
        try:
            next_id = next(id_iter)
            if next_id == "":
                next_id = 999
        except StopIteration:
            next_id = 999
        return f"[{next_id}"
    
    output = re.sub(r"\[", get_id_string, sentence)

    return output

def postprocess_result(sentence: str, result: str) -> str:
    """Apply several postprocessing functions to the result of the model
    to prepare it for the scoring functions.

    Args:
        sentence (str): target sentence without the identifiers for the mentions.
        result (str): result of the model. It might have slight differences with
        respect to the sentence and it might have mentions without identifiers.

    Returns:
        str: result of the model with identifiers for all the mentions and the
        text identical to the sentence.
    """
    # Discard the prompt from the output
    result = result.split("[/INST]")[-1]

    result = fill_identifiers_in_sentence(sentence, result)

    return result