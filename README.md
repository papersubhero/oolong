# Supplementary Materials of Oolong Evaluation

This repo contains supplementary materials of the evaluations on Oolong. 

## Environment Setup

```
conda env create -f oolongenv.yml
conda activate oolongenv
```

#### Discop Adaptation

We extended Discop code to support Qwen model, decompress the file in `adapted_discop/` to a folder named `Discop/`:
```
tar -xzf adapted_discop/adapted_discop.tar.gz -C Discop/
``` 

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

All the models follow the same naming convention:
- `fc`: trained with two fully connected NN
- `en/zh`: trained with EN or ZH data
- `len10`: all conversation data used in the training or testing of this model contain at least with 10 messages
- `dict`: trained using dictionary instantiation
- `discop`: trained using default Transfo model in Discop
- `qwen_discop`: trained using adapted Qwen model in Discop
- `probstrX11431`: stego rate set as 1 for `X=10`, 0.5 for `X=5`, 0.2 for `X=2`, 0.1 for `X=1`, 0.05 for `X=05`
- `ctxK`: context window size as 3 for `K=3`, 6 for `K=6`, 9 for `K=9`
- `epN`: with early stop mechanism for optimal model, the training stoped at epoch N
- `bsL`: batch size set to L

Models are in the corresponding subfolders:
- `fc_en_dict_models`
- `fc_en_qwen_models`
- `fc_en_tranfo_models`

## Useful Scripts

We provide the scripts in the subfolder `scripts/`.

#### Testing Model

The script `test_model.py` contains code to reproduce the experiment results using our pretrained models. 
There are several configurations (marked with `TOCONFIG`) for customization:
- `WEBP_FROM_TGS_DIR`: since CLIP-like models cannot process TGS files, we must convert the sticker files from TGS to WEBP files before computing the embedding. Our pretrained model and embedding dataset have contained precomputed results, so for reproducing purpose, there is no need of double converstion. 
- `class ContextClassifier`: we provide two classifier class definitions, one for 2FC, the other for LSTM, choose the corresponding classifier for the running test and comment out the other one
- `MSG/MEDIA_EMB_DB`: add the embedding DB path
- `test_dir_list`: put test data dir
- `model_list`: put the models to test, each model will be tested on ALL test dir in the previous list
- `media`: put the media universe file here for reference
- `BASE/PATH`: base path prepended before the test dir or model names
- `context_length`: context window size, consistent with the one of tested models, should be 3, 6, or 9

After configuration, run the test script to get precision, recall, f1 score. 
Sometimes the test script can take time to run, we recommend using `nohup` to run tests in the background and redircting terminal output into a file for later checking:
```bash
nohup python test.py > MODEL_NAME_TEST_DIR_NAME_TIMESTAMP.out 2>&1 &
```
