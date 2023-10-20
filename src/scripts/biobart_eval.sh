DEVICE_NUMBER=2
DATA_DIR=bigbio/data/no_abbr_res
PRETRAIN_PATH=kb_guided_pretrain_ckpt_hf
MAX_LEN=256

for DATASET in "gnormplus" "medmentions_st21pv" "ncbi_disease" "bc5cdr" "nlmchem" "medmentions_full" "nlm_gene" 
do
    CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                                $DATA_DIR/$DATASET \
                                                -evaluation \
                                                -model_load_path ./bigbio/models_checkpoint/${DATASET}_biobart/checkpoint-20000 \
                                                -model_token_path "GanjinZero/biobart-v2-large" \
                                                -trie_path $DATA_DIR/$DATASET/biobart_trie.pkl \
                                                -dict_path $DATA_DIR/$DATASET/target_kb.json \
                                                -per_device_eval_batch_size 1 \
                                                -prompt_tokens_enc 0 \
                                                -prompt_tokens_dec 0 \
                                                -prefix_prompt \
                                                -seed 0 \
                                                -num_beams 20 \
                                                -max_length 384 \
                                                -max_position_embeddings 1024 \
                                                -min_length 1 \
                                                -dropout 0.1 \
                                                -attention_dropout 0.1 \
                                                -testset \
                                                -prefix_mention_is 
                                                -output_path $DATA_DIR

done

# for DATASET in "ncbi_disease" "bc5cdr" "nlmchem"
# do
#     CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
#                                                 $DATA_DIR/$DATASET \
#                                                 -evaluation \
#                                                 -model_load_path ./bigbio/models_checkpoint/${DATASET}_biobart/checkpoint-20000 \
#                                                 -model_token_path "GanjinZero/biobart-v2-large" \
#                                                 -trie_path $DATA_DIR/$DATASET/biobart_trie.pkl \
#                                                 -dict_path $DATA_DIR/$DATASET/target_kb.json \
#                                                 -per_device_eval_batch_size 1 \
#                                                 -prompt_tokens_enc 0 \
#                                                 -prompt_tokens_dec 0 \
#                                                 -prefix_prompt \
#                                                 -seed 0 \
#                                                 -num_beams 20 \
#                                                 -max_length 384 \
#                                                 -max_position_embeddings 1024 \
#                                                 -min_length 1 \
#                                                 -dropout 0.1 \
#                                                 -attention_dropout 0.1 \
#                                                 -testset \
#                                                 -prefix_mention_is 

# done