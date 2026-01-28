#!/usr/bin/env python
# coding: utf-8

# torch related imports
import torch 
# import torchvision 
# print(torch.__version__)        # Should print 2.3.1+cu121
# print(torchvision.__version__)  # Should print 0.18.1+cu121
# print(torch.version.cuda)       # Should print 12.1
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
# clear GPU cache to be on the safe side of GPU run out of memory
import gc

import pandas as pd
import numpy as np

import os
# force pytorch to throw error exactly when happening, easier to debug
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# to find all csv files in dir
import glob

from PIL import features, Image
# print(features.check('webp'))  # Should print: True
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from pathlib import Path # for file name matching of TGS <-> WEBP
import time  # to measure training time
from datetime import datetime  # for timestamped filename
import sys  # for redirecting stdout
from contextlib import redirect_stdout  # for logging output
import wandb # monitor model via wandb
from wandb.sklearn import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# -------- CONFIGURATION --------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# try to use cuda 1 to avoid memory issue if cuda 1 is busy
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
# CLIP only support processing certain formats 
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# all types ["none", "image", "video", "sticker", "link", "other"]
MEDIA_TYPE_MAP = {'none': 0,'image': 1,'video': 2,'sticker': 3,'link': 4,'other': 5}

# db related import
import pickle # embedding flatten
import hashlib # db key computation 
from tqdm import tqdm # help visualize progress
import sqlite3 # for store and lookup cached embeddings

# for tgs compatibility with PIL, note that rlottie_python and pillow has conflict, use separate environment to pre-convert tgs to webp
# TOCONFIG change the mapping dir of tgs-webp
WEBP_FROM_TGS_DIR = "/PATH/TO/MAPPING/TGS/TO/WEBP/"

# ---- Hash Function ----
def blake2s_hash(text: str):
    # blake2s is as efficient as md5 while more collision resistant
    return hashlib.blake2s(text.encode('utf-8')).hexdigest()

# ---- SQLite Utility Functions ----
def init_db(path):
    conn = sqlite3.connect(path) # create the db if not exist yet, or load it from path
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                        hash TEXT PRIMARY KEY,
                        vector BLOB)''')
    conn.commit()
    return conn

def insert_embedding(conn, hash_key, embedding_tensor):
    cursor = conn.cursor()
    blob = pickle.dumps(embedding_tensor)
    cursor.execute("INSERT OR IGNORE INTO embeddings (hash, vector) VALUES (?, ?)", (hash_key, blob))
    conn.commit()

def get_embedding(conn, hash_key):
    cursor = conn.cursor()
    cursor.execute("SELECT vector FROM embeddings WHERE hash=?", (hash_key,))
    row = cursor.fetchone()
    if row:
        return pickle.loads(row[0])
    return None

# -------- DATASET --------
class ChatContextDataset(Dataset):
    def __init__(self, data_dir, context_length, media_folder, msg_emb_db_path, media_emb_db_path):
        self.samples = []  # stores rolling windows of conversation, each of length context_length
        self.context_length = context_length
        self.media_folder = media_folder  # store media_folder for later reconstruct full path

        # initialize DB connections
        self.msg_conn = init_db(msg_emb_db_path)
        self.media_conn = init_db(media_emb_db_path)

        all_files = glob.glob(os.path.join(data_dir, '*.csv'))
        for csv_path in all_files:
            df = pd.read_csv(csv_path)
            required_cols = ['sender_id', 'msg', 'media_type', 'media_location', 'label']
            if not all(col in df.columns for col in required_cols):
                print(f"[WARNING] Skipping {csv_path} due to missing required columns.")
                continue

            for i in range(0, len(df) - context_length + 1):
                window = df.iloc[i:i+context_length]
                self.samples.append(window)

    # context manager for persistent access
    def __enter__(self):
        return self  # for use in `with` block

    # context manager for safe closure
    def __exit__(self, exc_type, exc_value, traceback):
        self.msg_conn.close()
        self.media_conn.close()
        print("[INFO] Closed SQLite DB connections.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_df = self.samples[idx]
        embeddings = []

        for _, row in context_df.iterrows():
            msg_str = str(row['msg'])

            # hash and attempt to fetch from DB
            msg_hash = blake2s_hash(msg_str)
            msg_emb = get_embedding(self.msg_conn, msg_hash)

            if msg_emb is None:
                # truncate is set to be TRUE because CLIP does not support longer than 77 tokens by default
                msg_emb = CLIP_MODEL.encode_text(clip.tokenize(msg_str, truncate=True).to(DEVICE)).float().cpu() 
                insert_embedding(self.msg_conn, msg_hash, msg_emb)  # insert newly seens msg embedding to DB 

            media_filename = str(row['media_location'])
            if media_filename.lower() == 'nan' or media_filename.strip().lower() == 'none':
                media_path = 'none' # for NaN media_location, just assign "none", later it will convert to all-zero embedding
            else:
                media_path = os.path.join(self.media_folder, media_filename) # build full media path from media_folder

            ext = os.path.splitext(media_path)[-1].lower() # get extension to gracefully handle files like .tgs, not supported by PIL

            # hash and check from media DB
            media_hash = blake2s_hash(media_filename)
            media_emb = get_embedding(self.media_conn, media_hash)

            if media_emb is None:
                # handle non-seen media file and add to db
                if media_path != 'none' and os.path.isfile(media_path) and ext in MEDIA_EXTENSIONS:
                    try:
                        img = Image.open(media_path).convert("RGB")
                        img_tensor = CLIP_PREPROCESS(img).unsqueeze(0).to(DEVICE)
                        # by default the media_emb is float16, while the train_model function expects float32
                        # so we must conver the original embedding output by adding .float() at the end
                        media_emb = CLIP_MODEL.encode_image(img_tensor).float().cpu()  # move to CPU for caching
                        insert_embedding(self.media_conn, media_hash, media_emb)  # insert to DB
                    except Exception as e:
                        print(f"[ERROR] Failed to load image {media_path}: {e}")
                        media_emb = torch.zeros_like(msg_emb)
                elif media_path != 'none' and ext == ".tgs":  # tgs is not natively CLIP-able but we pre-converted them
                    try:
                        # find the converted webp from WEBP_FROM_TGS_DIR
                        webp_str = Path(media_path).stem+".webp"
                        media_path = os.path.join(WEBP_FROM_TGS_DIR, webp_str)
                        # process the corresponding webp 
                        img = Image.open(media_path).convert("RGB")
                        img_tensor = CLIP_PREPROCESS(img).unsqueeze(0).to(DEVICE) 
                        media_emb = CLIP_MODEL.encode_image(img_tensor).float().cpu() 
                    except Exception as e:
                        print(f"[ERROR] Failed to load .tgs file {media_path}: {e}")
                        media_emb = torch.zeros_like(msg_emb)
                else:
                    if media_path != 'none':
                        print(f"[WARNING] Skipping unsupported media type: {media_path}")
                    media_emb = torch.zeros_like(msg_emb)

            # the largest sender_id is 4,293,984,957, need normalization before converting to tensor directly
            sender_id = int(row['sender_id']) / 5_000_000_000  # Normalize to [0, 1]
            # use [[]] bc we need two-dimention tensor to fit the msg/media embedding
            # sender_vec = torch.tensor([[sender_id]], device=DEVICE).float() --> keep emb on cpu
            sender_vec = torch.tensor([[sender_id]]).float()

            # media_type is mapped using predefined MEDIA_TYPE_MAP
            media_idx = MEDIA_TYPE_MAP.get(str(row['media_type']), 0)
            # use [[]] bc we need two-dimention tensor to fit the msg/media embedding
            # media_vec = torch.tensor([[media_idx]], device=DEVICE).float() --> keep emb on cpu
            media_vec = torch.tensor([[media_idx]]).float()

            combined = torch.cat([msg_emb, media_emb, sender_vec, media_vec], dim=1).float()
            embeddings.append(combined.squeeze(0))

        x = torch.stack(embeddings).float()  # Ensure float32 type

        # groundtruth label is 1 if any row has 'circum'
        if (context_df['label'] == 'circum').any():
            label = 1
        else:
            label = 0

        return x, torch.tensor(label, dtype=torch.long)

# -------- MODEL --------
import torch.nn.functional as F

class ContextClassifier(nn.Module):
    def __init__(self, input_dim, context_length):
        super().__init__()
        self.flattened_dim = input_dim * context_length  # flatten full context window
        self.fc1 = nn.Linear(self.flattened_dim, self.flattened_dim)
        self.fc2 = nn.Linear(self.flattened_dim, 2)  # output logits for 2 classes

    def forward(self, x):
        # x shape: [batch_size, context_length, input_dim]
        x = x.view(x.size(0), -1)  # flatten to [batch_size, context_length * input_dim]
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # final output shape: [batch_size, 2]

#### TOCONFIG if the model is not 2FC but LSTM, use the corresponding classifier class below
# class ContextClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 2)

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         return self.fc(hn.squeeze(0))

# TOCONFIG precomputed embedding db path
MSG_EMB_DB = "/PATH/TO/EMBEDDING/DB/MSG.db"
MEDIA_EMB_DB = "/PATH/TO/EMBEDDING/DB/MEDIA.db"

def test_model(test_data_dir, media_folder, context_length, model_path, batch_size=8, msg_emb_db_path=MSG_EMB_DB, media_emb_db_path=MEDIA_EMB_DB):  
    # ---- Use context manager for dataset (DB connection closes automatically) ----
    with ChatContextDataset(test_data_dir, context_length, media_folder, msg_emb_db_path, media_emb_db_path) as test_dataset:  
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)  # pin_memory for efficiency

        # --- Load model architecture and weights ---
        embedding_dim = CLIP_MODEL.encode_text(clip.tokenize("test").to(DEVICE)).shape[-1]  # compute embedding size for model using a test string 
        input_dim = 2 * embedding_dim + 2  # msg embedding, media embedding, sender_id, media_type
        model = ContextClassifier(input_dim=input_dim, context_length=context_length).to(DEVICE) # pass context len to flat the embeddings
        # model = ContextClassifier(input_dim=input_dim, hidden_dim=256).to(DEVICE)  # initialize model
        print(f"[INFO] Loading model from {model_path}")  # start loading
        model.load_state_dict(torch.load(model_path)) 
        print(f"[INFO] Loaded model from {model_path}")  # loading confirmation

        model.eval()
        correct = total = test_loss = 0
        criterion = nn.CrossEntropyLoss()

        all_preds = []  # collect all predictions
        all_labels = []  # collect all ground truth labels
        all_probs = []  # collect predicted probability for ROC/AUC

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE, non_blocking=True).float()
                y = y.to(DEVICE, non_blocking=True)
                output = model(x)
                loss = criterion(output, y)
                test_loss += loss.item()
                probs = torch.softmax(output, dim=1)[:, 1]  # get probability of class=1
                pred = output.argmax(dim=1)
                all_probs.extend(probs.cpu().numpy())  # sklearn requires numpy for ROC computation
                all_preds.extend(pred.cpu().tolist())  # collect predictions
                all_labels.extend(y.cpu().tolist())  # collect ground truth labels
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = correct / total

        # compute label distribution
        label_1_count = sum(1 for label in all_labels if label == 1)
        label_0_count = sum(1 for label in all_labels if label == 0)
        total_windows = len(all_labels)
        percent_1 = (label_1_count / total_windows) * 100 if total_windows > 0 else 0

        # load sklearn libraries to compute evaluations
        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score

        # compute precision, recall, f1, auc
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) # ROC curve is misleading when data imbalanced

        # precision-recall auc is more informative for imbalanced data
        from sklearn.metrics import precision_recall_curve, average_precision_score
        pr_auc = average_precision_score(all_labels, all_probs)  # equivalent to PR AUC
        precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)

        print(f"Total context windows: {total_windows}")  
        print(f"Label 1 count: {label_1_count} ({percent_1:.2f}%)")  
        print(f"Label 0 count: {label_0_count}")  
        print(f"Test Loss = {test_loss / len(test_loader):.4f}, Accuracy = {test_acc:.2%}")
        print(f"Precision = {precision:.2%}, Recall = {recall:.2%}, F1 = {f1:.2%}, AUC = {auc:.4f}, PR AUC (average precision): {pr_auc:.4f}")  

        # print full classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["noncircum", "circum"]))

    return all_labels, all_probs, precision_vals, recall_vals  # return for ROC AUC and PR AUC curve plotting


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc(labels, probs, title):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on "+title)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    plt.close()

def plot_pr_curve(precision, recall, title):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve of "+title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

#### TOCONFIG insert the test data dir, can be multiple ones 
test_dir_list = [
    "TEST_DIR_ONE",
    "TEST_DIR_TWO"
]

# TOCONFIG put models to test here, note that the models should have consistent context window size across train and test.
# The context window size indicated in the training process of model should be the same as the one defined when invoking the test function, see comment below on the parameter context_length=X
model_list = [
    "MODELA.pth",
    "MODELB.pth",
    "MODELC.pth"
]

# TOCONFIG media universe path
media = "/PATH/TO/MEDIA_UNIVERSE"

for test_dir_name in test_dir_list:
    #### TOCONFIG prepend base paths before the test dir and model
    test = "/BASE/PATH/"+test_dir_name
    for model_name in model_list:
        model = "/BASE/PATH/"+model_name

        print(f"======== Testing on {test_dir_name} started at {time.ctime()} ========")
        res_labels, res_probs, res_precisions, res_recalls = test_model(
            test_data_dir=test,
            media_folder=media,
            context_length=X, # TOCONFIG change X to the corresponding context window size as the models, 3, 6, or 9
            model_path=model
        )
        # only output numbers, do not plot in script
        # plot_pr_curve(res_precisions, res_recalls, "Plot Title")
        print(f"======== Testing on {test_dir_name} ended at {time.ctime()} ========")

