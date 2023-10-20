#!/bin/bash

# for DATASET in "bc5cdr" "medmentions_full" "medmentions_st21pv" "gnormplus" "nlmchem" "nlm_gene" "ncbi_disease"
# for DATASET in "gnormplus"
# do
#     python ./trie/create_trie_and_target_kb.py --data_dir ./bigbio/data/$DATASET/ --use_biobart
# done

for DATASET in "medmentions_st21pv"  "gnormplus" "nlmchem" "nlm_gene" "ncbi_disease" "medmentions_full" "bc5cdr" 
# for DATASET in "gnormplus"
do
    python ./trie/create_trie_and_target_kb.py --data_dir ./bigbio/data/no_abbr_res/$DATASET
done

