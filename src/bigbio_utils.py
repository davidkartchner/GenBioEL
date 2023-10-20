import warnings
from collections import defaultdict
import pandas as pd
from bigbio.dataloader import BigBioConfigHelpers


conhelps = BigBioConfigHelpers()


dataset_names = [
    "bc5cdr",
    "medmentions_full",
    "medmentions_st21pv",
    "gnormplus",
    "nlmchem",
    "nlm_gene",
    "ncbi_disease",
    # 'plantnorm',
    # 'bc6id',
]

dataset_to_pretty_name = {
    "medmentions_full": "MedMentions Full",
    "medmentions_st21pv": "MedMentions ST21PV",
    "bc5cdr": "BC5CDR",
    "gnormplus": "GNormPlus",
    "ncbi_disease": "NCBI Disease",
    "nlmchem": "NLM Chem",
    "craft": "CRAFT",
    "bc6id": "BC6ID",
    "bc3gm": "BC3GM",
    "plantnorm": "PlantNorm",
    "nlm_gene": "NLM Gene",
}

model_to_color = {}


def cache_deduplicated_dataset(deduplicated_df):
    """
    Cache deduplicated dataset file for faster loading
    """
    raise NotImplementedError


def check_if_cached(dataset_name):
    """
    Check if dataset has been preprocessed and cached
    """
    raise NotImplementedError


def load_cached_dataset(dataset, splits_to_include: list = None):
    """
    Load cached deduplciated dataset as dataframe
    """
    raise NotImplementedError


def dataset_to_documents(dataset):
    """
    Return dictionary of documents in BigBio dataset
    """
    docs = {}
    data = {}
    for split in dataset.keys():
      print(split)
      type_split = {}
      for doc in dataset[split]:
            doc_id = pmid = doc["document_id"]
            doc_text = "\n".join([" ".join(x["text"]) for x in doc["passages"]])
            print(doc)
            break
            docs[doc_id] = doc_text
            type_split[doc_id] = doc_text
      data[split] = type_split
    print(data['test'])
    return docs


def dataset_to_df(dataset, splits_to_include: list = None):
    """
    Convert BigBio dataset to pandas DataFrame

    Params:
    ------------------
        dataset: BigBio Dataset
            Dataset to load from BigBio

        splits_to_include: list of str
            List of splits to include in mo
    """
    columns = [
        # 'context', # string
        "document_id",  # string
        "mention_id",  # string
        "text",  # string
        "type",  # list
        "offsets",  # list of lists
        # "db_name",
        "db_ids",  # list
        "split",  # string
    ]
    all_lines = []

    if splits_to_include is None:
        splits_to_include = dataset.keys()

    for split in splits_to_include:
        if split not in dataset.keys():
            warnings.warn(f"Split '{split}' not in dataset.  Omitting.")
        for doc in dataset[split]:
            pmid = doc["document_id"]
            for e in doc["entities"]:
                if len(e["normalized"]) == 0:
                    continue
                text = " ".join(e["text"])
                offsets = ";".join(
                    [",".join([str(y) for y in x]) for x in e["offsets"]]
                )
                # db_name = e["normalized"][0]["db_name"]
                db_ids = [x["db_name"] + ":" + x["db_id"] for x in e["normalized"]]
                all_lines.append(
                    [
                        pmid,
                        e["id"],
                        text,
                        e["type"],
                        # e['offsets'],
                        offsets,
                        # db_name,
                        db_ids,
                        split,
                    ]
                )

    df = pd.DataFrame(all_lines, columns=columns)

    deduplicated = (
        df.groupby(["document_id", "offsets"])
        .agg(
            {
                "text": "first",
                "type": lambda x: list(set([a for a in x])),
                "db_ids": lambda db_ids: list(set([y for x in db_ids for y in x])),
                "split": "first",
            }
        )
        .reset_index()
    )

    deduplicated["offsets"] = deduplicated["offsets"].map(
        lambda x: [y.split(",") for y in x.split(";")]
    )

    return deduplicated


def resolve_abbreviation(document_id, text, abbreviations_dict):
    """
    Return un-abbreviated form of entity name if it was found in abbreviations_dict, else return original text

    Inputs:
    -------------------------------
        document_id: str
            ID of document where mention was found

        text: str
            Text of mention

        abbreviations_dict: dict
            Dict of form {document_id:{text: unabbreviated_text}} containing abbreviations detected in each document
    """
    if text in abbreviations_dict[document_id]:
        return abbreviations_dict[document_id][text]
    else:
        return text


def metamap_text_to_candidates(metamap_output):
    """
    Create mapping from text to list of candidates output by metamap

    Inputs:
    -------------------------------
        filepath: string or path-like
            Path to file containing output of MetaMap

    Returns:
    -------------------------------
        text2candidates: dict
            Dict of form {mention_text: [cand1, cand2, ...]} mapping each text string to candidates
            generated by metamap.  If no candidates were generated for a given key, value will be
            an empty string.
    """
    text2candidates = defaultdict(list)

    for row in metamap_output[
        ["text", "mapping_cui_list", "candidate_cui_list"]
    ].values:
        text = row[0]
        candidates = eval(row[1]) + eval(row[2])

        # TODO: Need to account for correct database
        raise NotImplementedError("Need to map UMLS values to correct DB")
        candidates = ["UMLS:" + x for x in candidates]

        text2candidates[text] = candidates
    return text2candidates


if __name__ == "__main__":
    dataset_name = "bc5cdr"
    dataset = conhelps.for_config_name(f"{dataset_name}_bigbio_kb").load_dataset()
    dataset_to_documents(dataset)
