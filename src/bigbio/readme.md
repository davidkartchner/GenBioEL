# How to run our code:

# To create necessary data files for the model

Add the entity mappings as a txt file to corresponding dataset folder with dataset name Eg: bc5cdr.txt
Now, run biogenel_bigbioloader.py.
This python file will create .source,.target files for train, test, dev data and target_kb.json.

# Create a trie
Once data is generated create a trie using the instructions in the other readme.

# To finetune and eval the model

Run the bash files bigbio_train_${datasetname}.sh by adding dataset path, pretrained path.


