#!/bin/bash

bash scripts/finetune_all_models.sh 
echo "DONE FINETUNING!!!"

bash scripts/eval_all_models.sh
echo "DONE EVALUATING!!!"