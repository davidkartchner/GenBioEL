DEVICE_NUMBER=$1
DATA_DIR=bigbio/data/no_abbr_res
PRETRAIN_PATH=kb_guided_pretrain_ckpt_hf
MAX_LEN=256
# MODEL_SAVE_PATH = ./bigbio/models_checkpoint/$DATASET

# for DATASET in "medmentions_st21pv" "medmentions_full" # "nlm_gene" "ncbi_disease"
for DATASET in "bc5cdr" "gnormplus" "nlmchem" "nlm_gene" "ncbi_disease" 
# for DATASET in "medmentions_st21pv"
do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                                $DATA_DIR/$DATASET \
                                                -evaluation \
                                                -model_load_path ./bigbio/models_checkpoint/$DATASET/checkpoint-20000 \
                                                -model_token_path facebook/bart-large \
                                                -trie_path $DATA_DIR/$DATASET/trie.pkl \
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

done


# for DATASET in "medmentions_st21pv"
# do
#     CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
#                                                 $DATA_DIR/$DATASET \
#                                                 -evaluation \
#                                                 -model_load_path ./bigbio/models_checkpoint/$DATASET/checkpoint-50000 \
#                                                 -model_token_path facebook/bart-large \
#                                                 -trie_path $DATA_DIR/$DATASET/trie.pkl \
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
