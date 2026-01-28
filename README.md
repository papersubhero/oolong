# Supplementary Materials of Oolong Evaluation

This repo contains supplementary materials of the evaluations on Oolong. 

## Environment Setup

```
conda env create -f oolongenv.yml
conda activate oolongenv
```

#### Discop Adaptation

We extended Discop code to support Qwen model in `qwen_discop/Discop`. 

#### Conversation Data Embedding

For efficient evaluation, we precompute the embeddings of the conversation data and store the embeddings in datasets. 

- `en_msg_emb.db` (386M) contains CLIP-computed embeddings for all `msg` strings in dictionary EN conversation dataset
- `en_media_emb.db` (109M) contains CLIP-computed embeddings for all `media` files in dictionary EN conversation dataset
- `en_discop_msg_emb.db` (1.8G) contains CLIP-computed embeddings for all `msg` strings in Discop EN conversation dataset
- `en_discop_media_emb.db` (433M) contains CLIP-computed embeddings for all `media` files in Discop EN conversation dataset 
- `zh_all_msg_emb.db` (723M) contains Chinese-CLIP-computed embeddings for all `msg` strings in Chinese (ZH) conversation dataset
- `zh_all_media_emb.db` (10M) contains Chinese-CLIP-computed embeddings for all `media` files in ZH conversation dataset

For access control of the data embeddings, we encrypted all the databases using `age`, and share the key only for review purpose. 
To decrypt each database, fill in the correct full path of key file, the name of database, and run: 
```
age -d -i PATH/TO/KEY/FILE/key.txt -o emb_db/DBNAME.db emb_db/DBNAME.db.age
```
## Ready-to-Test Models

To reproduce the experiment results in our evaluation section, we provide the following pre-trained models.

#### EN Dictionary Models
- 

#### EN Discop Models

#### ZH Dictionary Models

#### ZH Discop Models

## Useful Scripts

We provide the following scripts that were used in our evaluation so that future researchers can adapt to other projects. 
- test script
- train script
- embedding computation 

