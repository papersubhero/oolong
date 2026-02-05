# Supplementary Materials of Oolong Evaluation

This repo contains supplementary materials of the evaluations on Oolong. 
We describe the full pipeline of environment setup, data generation, training, and testing. 
To reproduce our experiment results, we also provide pretrained models with the corresponding test data. 

## Environment Setup

```
conda env create -f oolongenv.yml
conda activate oolongenv
```

## Data Generation of Instantiations

### Dictionary Instantiation

To generate train/test dataset using the dictionary instantiation, run:

```bash
python3 scripts/dictionary_instantiation.py 
```
This python script contains several configurations (marked with `TOCONFIG`) for customization:

- training parameters such as `train_dataset_path` the input path to original conversation data, `train_output_path` the output path to training date folder, `train_link_universe_path` the path to all links in the stego space, `train_media_universe_path` the path to all media files in the stego space

- testing parameters, other than the path parameters that are similar to the training ones, `test_circum_population_rate` is to set the percentage of circumventing conversations among all conversations

- a list of stego rates that determine the percentage of stego messages among all messages

### Discop Instantiation

To generate train/test dataset using the Discop instantiation, run:

```bash
python scripts/discop_instantiation.py 
```
This python script contains several configurations (marked with `TOCONFIG`) for customization:

- Discop source code path

- length of embedded sensitive information in bit, the function`get_stego_discop` can be customized to generate messages embedded with a certain amount of bits with the corresponding length of output

- training parameters such as `train_dataset_path` the input path to original conversation data, `train_output_path` the output path to training date folder

- testing parameters, other than the path parameters that are similar to the training ones, `test_circum_population_rate` is to set the percentage of circumventing conversations among all conversations

- a list of stego rates that determine the percentage of stego messages among all messages

#### Discop Adaptation

We extended Discop code to support Qwen2.5-0.5B model, decompress the file in `adapted_discop/` to a folder named `Discop/`:
```
tar -xzf adapted_discop/adapted_discop.tar.gz -C Discop/
``` 

## Detector Training

Training a detector can take time, we recommend using `nohup` to run scripts in the background and redircting terminal output into a file for later checking:
```bash
nohup python scripts/train.py > MODEL_NAME_TRAIN_DIR_NAME_TIMESTAMP.out 2>&1 &
```
This python script contains several configurations (marked with `TOCONFIG`) for customization:

- path to pre-converted TGS files to WEBP `WEBP_FROM_TGS_DIR`, since CLIP-like models cannot process TGS files, we must convert the sticker files from TGS to WEBP files before computing the embedding. Our pretrained model and embedding dataset have contained precomputed results, so for reproducing purpose, there is no need of double converstion

- classifier models, e.g., 2FC, LSTM in `class ContextClassifier`
- path to pre-computed embeddings `MSG_EMB_DB`, `MEDIA_EMB_DB`
- path to backup models `BACKUP_DIR` in each epoch
- context window size `context_length`, e.g., 3, 6, or 9
- path to training data folder `train_list`, can be a list of folders


## Detector Testing

After training, run the test script to get precision, recall, f1 score. 
Sometimes the test script can also take time to run, we recommend using `nohup` to run tests in the background and redircting terminal output into a file for later checking:
```bash
nohup python scripts/test.py > MODEL_NAME_TEST_DIR_NAME_TIMESTAMP.out 2>&1 &
```

There are several configurations (marked with `TOCONFIG`) for customization:
- similar parameters to the training ones: `WEBP_FROM_TGS_DIR`, `class ContextClassifier`, `MSG/MEDIA_EMB_DB`
- path to testing data folder `test_dir_list`, can be a list of folders
- path to `model_list`: put the models to test, each model will be tested on ALL test dir in the previous list
- context window size `context_length`, consistent with the one of tested models, e.g., 3, 6, or 9

## Reproducing Experiments

We further provide the test data together with the precomputed embedding and the precomputed models in our experiments. 

### **Data Encryption As Precaution**

For access control of the conversation data (and their embeddings), we encrypted all the database-related files using `age`, and share the key only for review purpose in the Open Science field on the HotCRP website. 
To decrypt each encrypted file with extension `.age`, first install `age` through command line:

```bash
conda install -c conda-forge age
```

After successful installation, fill in the correct full path of key file, the name of database, and run: 

```bash
age -d -i PATH/TO/KEY/FILE/key.txt -o emb_db/DBNAME.db emb_db/DBNAME.db.age
```

### Conversation Data Embedding

For efficient evaluation, we precompute the embeddings of the conversation data and store the embeddings in datasets. 

- `en_msg_emb.db` (386M) contains CLIP-computed embeddings for all `msg` strings in dictionary EN conversation dataset
- `en_media_emb.db` (109M) contains CLIP-computed embeddings for all `media` files in dictionary EN conversation dataset
- `en_discop_msg_emb.db` (1.8G) contains CLIP-computed embeddings for all `msg` strings in Discop EN conversation dataset
- `en_discop_media_emb.db` (433M) contains CLIP-computed embeddings for all `media` files in Discop EN conversation dataset 
- `zh_all_msg_emb.db` (723M) contains Chinese-CLIP-computed embeddings for all `msg` strings in Chinese (ZH) conversation dataset
- `zh_all_media_emb.db` (10M) contains Chinese-CLIP-computed embeddings for all `media` files in ZH conversation dataset

### Test Data

We provide the (encrypted) test data in `test_data` for review purpose as well as the corresponding media files in `all_media`, and we plan to release the training data as well upon approval from the conference ethics review board. 
The test data are encrypted the same way as the embeddings (and hence, can be decrypted in the same way as the embeddings):
- `en_dict`: test data in EN for dictionary instantiation
- `en_transfo_discop`: test data in EN for Transfo Discop instantiation
- `en_qwen_discop`: test data in EN for Qwen Discop instantiation
- `zh_dict`: test data in ZH for dictionary instantiation
- `zh_discop`: test data in ZH for both Transfo and Qwen Discop instantiation

### Ready-to-Test Models

To reproduce the experiment results in our evaluation section, we provide the following pre-trained models.

All the models follow the same naming convention:
- `fc/lstm`: trained with two fully connected NN (`fc`) or LSTM (`lstm`)
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


