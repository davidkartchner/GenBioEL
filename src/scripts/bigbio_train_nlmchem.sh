DEVICE_NUMBER=0
DATASET=bigbio/data
MODEL_NAME=nlmchem
PRETRAIN_PATH=kb_guided_pretrain_ckpt_hf
# MODEL_SAVE_PATH = ./bigbio/models_checkpoint/ncbi_disease

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/nlmchem \
                                            -model_load_path $PRETRAIN_PATH \
                                            -model_save_path ./bigbio/models_checkpoint/nlmchem \
                                            -model_token_path facebook/bart-large \
                                            -save_steps 20000 \
                                            -logging_path ./logs/$MODEL_NAME \
                                            -logging_steps 100 \
                                            -init_lr 3e-7 \
                                            -per_device_train_batch_size 8 \
                                            -evaluation_strategy no \
                                            -label_smoothing_factor 0.1 \
					                        -rdrop 0 \
					                        -gradient_accumulate 1 \
                                            -max_grad_norm 0.1 \
                                            -max_steps 20000 \
					                        -warmup_steps 0 \
                                            -weight_decay 0.01 \
                                            -lr_scheduler_type polynomial \
                                            -attention_dropout 0.1  \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -max_length 256 \
                                            -max_position_embeddings 1024 \
                                            -seed 0 \
                                            -prefix_mention_is \
                                            -finetune 

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/nlmchem \
                                            -evaluation \
                                            -model_load_path ./bigbio/models_checkpoint/nlmchem/checkpoint-20000 \
                                            -model_token_path facebook/bart-large \
                                            -trie_path $DATASET/ncbi_disease/trie.pkl \
					                        -dict_path $DATASET/ncbi_disease/target_kb.json \
                                            -per_device_eval_batch_size 1 \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -prefix_prompt \
					                        -seed 0 \
                                            -num_beams 16 \
                                            -max_length 256 \
                                            -max_position_embeddings 1024 \
                                            -min_length 1 \
                                            -dropout 0.1 \
                                            -attention_dropout 0.1 \
                                            -prefix_mention_is \
