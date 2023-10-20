DEVICE_NUMBER=$1
DATA_DIR=bigbio/data/no_abbr_res
PRETRAIN_PATH=kb_guided_pretrain_ckpt_hf
MAX_LEN=256
# MODEL_SAVE_PATH = ./bigbio/models_checkpoint/$DATASET

# for DATASET in "gnormplus" "ncbi_disease" "gnormplus"  "nlm_gene" "bc5cdr"  "medmentions_full" "medmentions_st21pv" "nlmchem"  
for DATASET in "medmentions_st21pv"

do
    echo "*** Start Training ***"
    date +%T
    CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                                $DATA_DIR/$DATASET \
                                                -model_load_path "GanjinZero/biobart-v2-large" \
                                                -model_save_path ./bigbio/models_checkpoint/${DATASET}_biobart/no_abbr_res \
                                                -model_token_path "GanjinZero/biobart-v2-large" \
                                                -save_steps 5000 \
                                                -logging_path ./logs/${DATASET}_biobart \
                                                -logging_steps 100 \
                                                -init_lr 3e-7 \
                                                -per_device_train_batch_size 8 \
                                                -evaluation_strategy no \
                                                -label_smoothing_factor 0.1 \
                                                -rdrop 0 \
                                                -gradient_accumulate 1 \
                                                -max_grad_norm 0.1 \
                                                -max_steps 20000 \
                                                -trie_path $DATA_DIR/$DATASET/biobart_trie.pkl \
                                                -warmup_steps 0 \
                                                -weight_decay 0.01 \
                                                -lr_scheduler_type polynomial \
                                                -attention_dropout 0.1  \
                                                -prompt_tokens_enc 0 \
                                                -prompt_tokens_dec 0 \
                                                -max_length 384 \
                                                -max_position_embeddings 1024 \
                                                -seed 0 \
                                                -prefix_mention_is \
                                                -finetune 
    echo "*** End Training ***"
    date +%T
                                            
done

