import os
import ujson
import warnings
from collections import defaultdict
import pandas as pd
from bigbio.dataloader import BigBioConfigHelpers
import json
import joblib
import numpy as np

from tqdm.auto import tqdm

tqdm.pandas()

import sys

sys.path.append("../../..")
from bigbio_utils import CUIS_TO_REMAP, CUIS_TO_EXCLUDE, DATASET_NAMES
from bigbio_utils import dataset_to_documents, dataset_to_df, resolve_abbreviation


conhelps = BigBioConfigHelpers()


def put_mention_in_context(
    doc_id,
    doc,
    mention,
    offsets,
    max_len,
    start_delimiter="START ",
    end_delimiter=" END",
    resolve_abbrevs=False,
    abbrev_dict=None,
):
    """
    Put a mention in context of surrouding text
    """
    # Resolve abbreviaions if desired
    if resolve_abbrevs and abbrev_dict is not None:
        mention = resolve_abbreviation(doc_id, mention, abbrev_dict)

    tagged_mention = start_delimiter + mention + end_delimiter
    start = offsets[0][0]
    end = offsets[-1][-1]
    before_context = doc[:start]
    after_context = doc[end:]
    before_split_context = before_context.split(" ")
    after_split_context = after_context.split(" ")
    len_before = len(before_split_context)
    len_after = len(after_split_context)

    max_ctx_l_len = max_len // 2
    max_ctx_r_len = max_len - max_ctx_l_len

    if len_before < max_ctx_l_len and len_after < max_ctx_r_len:
        mention_in_context = before_context + tagged_mention + after_context

    elif len_before >= max_ctx_l_len and len_after >= max_ctx_r_len:
        mention_in_context = (
            " ".join(before_split_context[-max_ctx_l_len:])
            + " "
            + tagged_mention
            + " "
            + " ".join(after_split_context[:max_ctx_r_len])
        )

    elif len_before >= max_ctx_l_len:
        ctx_l_len = max_len - len_after
        mention_in_context = (
            " ".join(before_split_context[-ctx_l_len:])
            + " "
            + tagged_mention
            + after_context
        )
    else:
        ctx_r_len = max_len - len_before
        mention_in_context = (
            before_context
            + tagged_mention
            + " "
            + " ".join(after_split_context[:ctx_r_len])
        )

    return mention_in_context


def contextualize_mentions(
    doc_dict,
    deduplicated,
    max_len=128,
    resolve_abbrevs=False,
    abbrev_dict=None,
    start_delimiter="START ",
    end_delimiter=" END",
):
    deduplicated["contextualized_mention"] = deduplicated[
        ["document_id", "mention", "offsets"]
    ].progress_apply(
        lambda x: put_mention_in_context(
            x[0],
            doc_dict[x[0]],
            x[1],
            x[2],
            max_len=max_len,
            resolve_abbrevs=resolve_abbrevs,
            abbrev_dict=abbrev_dict,
        ),
        axis=1,
    )

    return deduplicated


def create_training_files(
    save_dir: str,
    document_dict: dict,
    deduplicated,
    abbreviations_dict: dict,
    cui2alias: dict,
    resolve_abbrevs: bool=True,
):
    # Get contextualized mentions
    df = contextualize_mentions(
        document_dict,
        deduplicated,
        resolve_abbrevs=resolve_abbrevs,
        abbrev_dict=abbreviations_dict,
    )

    # df["entity_aliases"] = df["db_ids"].map(
    #     lambda x: list(set([z for y in x for z in cui2alias[y]]))
    # )
    # df["most_similar_alias"] = df[["mention", "entity_aliases"]].apply(
    #     lambda x: get_most_similar_alias(x[0], x[1], vectorizer)
    # )

    # Get closest synonym for each mention
    tfidf_vectorizer = "../tfidf_vectorizer.joblib"
    vectorizer = joblib.load(tfidf_vectorizer)

    df["entity_aliases"] = df["db_ids"].map(
        lambda x: list(set([z for y in x for z in cui2alias[y]]))
    )
    # print(df[df.entity_aliases.map(lambda x: len(x)==0)])
    print("Getting most similar alias")
    df["most_similar_alias"] = df[["mention", "entity_aliases"]].progress_apply(
        lambda x: get_most_similar_alias(
            mention=x[0], cui_alias_list=x[1], vectorizer=vectorizer
        ),
        axis=1,
    )

    # Make json string in correct format for decoder
    df["source_json"] = df["contextualized_mention"].map(lambda x: ujson.dumps([x]))

    df["target_json"] = df[["mention", "most_similar_alias"]].progress_apply(
        lambda x: ujson.dumps([f"{x[0]} is", x[1]]), axis=1
    )

    # Store/check results
    # print(df.head(5))
    # print("Saving Pickle)")
    # df.to_pickle(os.path.join(save_dir, "processed_mentions.pickle"))

    # Write data files to pickle

    for split in df.split.unique():
        subset = df.query("split == @split")
        split_name = split
        if split in ["valid", "validation"]:
            split_name = "dev"

        # Write files
        with open(os.path.join(save_dir, f"{split_name}label.txt"), "w") as f:
            entity_link_list = ["|".join(x) for x in subset.db_ids.tolist()]
            f.write("\n".join(entity_link_list))

        with open(os.path.join(save_dir, f"{split_name}.source"), "w") as f:
            f.write("\n".join(subset.source_json.tolist()))

        with open(os.path.join(save_dir, f"{split_name}.target"), "w") as f:
            f.write("\n".join(subset.target_json.tolist()))


def get_most_similar_alias(mention, cui_alias_list, vectorizer):
    """
    Get most similar CUI alias to current mention using TF-IDF
    """
    most_similar_idx = cal_similarity_tfidf(cui_alias_list, mention, vectorizer)[0]
    return cui_alias_list[most_similar_idx]


def cal_similarity_tfidf(a: list, b: str, vectorizer):
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    features_T = features_a.T
    sim = features_b.dot(features_T).todense()
    return sim[0].argmax(), np.max(np.array(sim)[0])


def create_target_kb_dict(file_path, dataset_name):
    with open(file_path + f"{dataset_name}_aliases.txt", "r") as f:
        lines = f.read().split("\n")

    # Construct dict mapping each CURIE to a list of aliases
    umls_dict = {}
    for line in tqdm(lines):
        if len(line.split("||")) != 2:
            print(line)
            continue
        cui, name = line.split("||")
        if cui in umls_dict:
            umls_dict[cui].add(name.lower())
        else:
            umls_dict[cui] = set([name.lower()])

    processed_dict = {key: list(val) for key, val in umls_dict.items()}
    return processed_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Specific dataset to evaluate on')

    args = parser.parse_args()

    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        datasets = [
            "ncbi_disease",
            "bc5cdr",
            "nlmchem",
            "medmentions_full",
            "medmentions_st21pv",
            "gnormplus",
            "nlm_gene",
        ]
        
    # datasets = ["medmentions_st21pv","medmentions_full"]
    for dataset_name in datasets:
        # read data

        save_dir = f"./data/no_abbr_res/{dataset_name}/"
        dataset = conhelps.for_config_name(f"{dataset_name}_bigbio_kb").load_dataset()

        target_kb_dict = create_target_kb_dict(save_dir, dataset_name)
        # create target_kb.json
        with open(os.path.join(save_dir, "target_kb.json"), "w") as target_kb:
            str_target = ujson.dumps(target_kb_dict, indent=2)
            target_kb.write(str_target)

        with open("./abbreviations.json") as json_file:
            abbreviations_dict = ujson.load(json_file)
        entity_remapping_dict = CUIS_TO_REMAP[dataset_name]
        entities_to_exclude = CUIS_TO_EXCLUDE[dataset_name]

        deduplicated = dataset_to_df(
            dataset,
            entity_remapping_dict=entity_remapping_dict,
            cuis_to_exclude=entities_to_exclude,
        )
        deduplicated["mention"] = deduplicated["text"]

        split_docs = dataset_to_documents(dataset)

        # create source file
        create_training_files(
            save_dir,
            document_dict=split_docs,
            deduplicated=deduplicated,
            abbreviations_dict=abbreviations_dict,
            cui2alias=target_kb_dict,
            resolve_abbrevs=False
        )
