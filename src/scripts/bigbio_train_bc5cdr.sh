DEVICE_NUMBER=0
DATASET=bigbio/data
MODEL_NAME=bc5cdr
PRETRAIN_PATH=kb_guided_pretrain_ckpt_hf

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                           $DATASET/bc5cdr \
                                           -model_load_path $PRETRAIN_PATH \
                                           -model_token_path facebook/bart-base \
                                           -model_save_path ./bigbio/models_checkpoint/bc5cdr\
                                           -save_steps 20000 \
                                           -logging_path ./logs/$MODEL_NAME \
                                           -logging_steps 100 \
                                           -init_lr 1e-05 \
                                           -per_device_train_batch_size 8 \
                                           -evaluation_strategy no \
                                           -label_smoothing_factor 0.1 \
                                           -max_grad_norm 0.1 \
                                           -max_steps 20000 \
					                        -warmup_steps 500 \
                                           -weight_decay 0.01 \
					                        -rdrop 0.0 \
                                           -lr_scheduler_type polynomial \
                                           -attention_dropout 0.1  \
                                           -prompt_tokens_enc 0 \
                                           -prompt_tokens_dec 0 \
                                           -max_length 256 \
                                           -max_position_embeddings 1024 \
                                           -seed 0 \
					                        -finetune \
					                        -prefix_mention_is 
                                            

CUDA_VISIBLE_DEVICES=$DEVICE_NUMBER python ./train.py \
                                            $DATASET/bc5cdr \
                                            -model_token_path facebook/bart-base \
                                            -evaluation \
					                        -dict_path $DATASET/bc5cdr/target_kb.json \
                                            -trie_path $DATASET/bc5cdr/trie.pkl  \
                                            -per_device_eval_batch_size 1 \
					                        -model_load_path ./bigbio/models_checkpoint/bc5cdr/checkpoint-20000\
					                        -seed 0 \
                                            -prompt_tokens_enc 0 \
                                            -prompt_tokens_dec 0 \
                                            -prefix_prompt \
                                            -num_beams 16 \
                                            -max_length 256 \
                                            -min_length 1 \
                                            -dropout 0.1 \
                                            -attention_dropout 0.1 \
                                            -prefix_mention_is \
