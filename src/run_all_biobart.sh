#!/bin/bash

echo "Finetuning BioBART"
bash scripts/biobart_finetune.sh

echo "Evaluating BioBART"
bash scripts/biobart_eval.sh
