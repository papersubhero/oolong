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

# cuda configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

#### TOCONFIG for tgs compatibility with PIL, note that rlottie_python and pillow has conflict, use separate environment to pre-convert tgs to webp
WEBP_FROM_TGS_DIR = "/PATH/TO/TGS/MAPPING/WEBP"

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
                # need to convert to float32, and move to cpu for storage
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
                        # somehow by default the media_emb is float16, while the train_model function expects float32
                        # so we need to conver the original embedding output by adding .float() at the end
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

            # TOCONFIG normalization using the largest possible sender_id before converting to tensor directly
            sender_id = int(row['sender_id']) / 5_000_000_000  # Normalize to [0, 1]
            # use [[]] bc we need two-dimention tensor to fit the msg/media embedding
            sender_vec = torch.tensor([[sender_id]]).float()

            # media_type is mapped using predefined MEDIA_TYPE_MAP
            media_idx = MEDIA_TYPE_MAP.get(str(row['media_type']), 0)
            # use [[]] bc we need two-dimention tensor to fit the msg/media embedding
            media_vec = torch.tensor([[media_idx]]).float()

            combined = torch.cat([msg_emb, media_emb, sender_vec, media_vec], dim=1).float()
            embeddings.append(combined.squeeze(0))

        x = torch.stack(embeddings).float()  # ensure float32 type
        
        # groundtruth label is 1 if any row has 'circum'
        if (context_df['label'] == 'circum').any():
            label = 1
        else:
            label = 0
        
        return x, torch.tensor(label, dtype=torch.long)

# -------- MODEL --------
import torch.nn.functional as F
# TOCONFIG choose classifier model, e.g., 2FC or LSTM

#### 2FC model
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

##### LSTM model
# class ContextClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 2)

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         return self.fc(hn.squeeze(0))

# TOCONFIG precomputed embedding db
MSG_EMB_DB = "/PATH/TO/MSG.db"
MEDIA_EMB_DB = "/PATH/TO/MEDIA.db"
# TOCONFIG save checkpoints as a backup
BACKUP_DIR = "/PATH/TO/BACKUP/FOLDER/"

# -------- TRAINING --------
# TOCONFIG check epoch number is enough (with early stop enabled), e.g., 50
def train_model(data_dir, media_folder, context_length, epochs=XX, batch_size=8, resume_path=None, msg_emb_db_path=MSG_EMB_DB, media_emb_db_path=MEDIA_EMB_DB): 
    # initialize wandb logging
    wandb.init(project="stegochat-detection", config={ 
        "context_length": context_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
    })
    # for better labelling of log and model
    data_dir_label = Path(data_dir).name
    # track training time
    start_time = time.time()  
    # create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"fc_log_{data_dir_label}_ctx{context_length}_ep{epochs}_bs{batch_size}_{timestamp}.txt"
    with open(log_filename, 'w') as f, redirect_stdout(f):  # redirect all print output to log file

        # ---- Use context manager for dataset (DB connection closes automatically) ----
        with ChatContextDataset(data_dir, context_length, media_folder, msg_emb_db_path, media_emb_db_path) as dataset: 
            val_size = int(0.1 * len(dataset))
            train_size = len(dataset) - val_size  
            train_set, val_set = random_split(dataset, [train_size, val_size])  # split into train and val
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True)

            
            # dynamically assign embedding dimension
            embedding_dim = CLIP_MODEL.encode_text(clip.tokenize("test").to(DEVICE)).shape[-1]
            input_dim = 2 * embedding_dim + 2  # msg embedding, media embedding, sender_id, media_type
            model = ContextClassifier(input_dim=input_dim, context_length=context_length).to(DEVICE) 

            # load model state if resume_path is provided
            if resume_path and os.path.isfile(resume_path):
                model.load_state_dict(torch.load(resume_path))
                print(f"[INFO] Resumed model loaded from {resume_path}")

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # -------- Early Stopping Initialization --------
            best_val_loss = float('inf')  # keep track of best validation loss
            patience = 3  # number of epochs with no improvement to tolerate
            epochs_without_improvement = 0  

            # # -------- In Case of Resume --------
            # resume = True  # set to True if you want to resume from checkpoint
            # start_epoch = 0

            # if resume:
            #     checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
            #     if os.path.exists(checkpoint_path):
            #         checkpoint = torch.load(checkpoint_path)
            #         model.load_state_dict(checkpoint['model_state_dict'])
            #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #         start_epoch = checkpoint['epoch'] + 1 # need to set epoch in range(start_epoch, num_epochs)
            #         print(f"Resuming from epoch {start_epoch}")
            #     else:
            #         print("Checkpoint not found. Starting from scratch.")

            for epoch in range(epochs):
                # --- Training ---
                model.train()
                total_loss = 0
                correct = total = 0
                for x, y in train_loader:
                    # x must be float32, y must be long
                    x = x.to(DEVICE, non_blocking=True).float()
                    y = y.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                
                train_acc = correct / total
                print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}, Accuracy = {train_acc:.2%}")
                wandb.log({ 
                    "epoch": epoch + 1,
                    "train_loss": total_loss / len(train_loader),
                    "train_acc": train_acc,
                })

                # --- Validation ---
                model.eval()
                correct = total = val_loss = 0
                # for ROC computation, confusion matrix and classifier report
                all_probs = []  
                all_preds = []
                all_labels = []  
                with torch.no_grad():
                    for x, y in val_loader:
                        # x must be float32, y must be long
                        x = x.to(DEVICE, non_blocking=True).float()
                        y = y.to(DEVICE, non_blocking=True)
                        output = model(x)
                        loss = criterion(output, y)
                        val_loss += loss.item()
                        # ROC requires prob
                        probs = torch.softmax(output, dim=1)[:, 1] 
                        all_probs.extend(probs.cpu().numpy()) 
                        all_labels.extend(y.cpu().numpy())  
                        # prediction label 0 or 1
                        pred = output.argmax(dim=1)
                        # for confusion matrix and classifier report
                        all_preds.extend(pred.cpu().numpy())
                        correct += (pred == y).sum().item()
                        total += y.size(0)
                val_acc = correct / total
                val_loss_avg = val_loss / len(val_loader) # init val_loss avg
                print(f"Validation Loss = {val_loss_avg:.4f}, Accuracy = {val_acc:.2%}")

                # log val metrics, confusion matrix, classifier repot and ROC
                cm = confusion_matrix(all_labels, all_preds)
                wandb.sklearn.plot_confusion_matrix(all_labels, all_preds, labels=[0,1])
                print("Confusion Matrix & Classifier Report:")
                print(cm)
                report_dict = classification_report(all_labels, all_preds, output_dict=True)
                for label, metrics in report_dict.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            wandb.log({f"class_{label}/{metric_name}": value})
                print(report_dict)
                auc = roc_auc_score(all_labels, all_probs)  
                fpr, tpr, _ = roc_curve(all_labels, all_probs)  
                wandb.log({  
                    "epoch": epoch + 1,
                    "val_loss": val_loss_avg,
                    "val_acc": val_acc,
                    "roc_auc": auc,
                    "fpr": fpr.tolist(), 
                    "tpr": tpr.tolist()
                    # "roc_curve": wandb.plot.roc_curve(all_labels, all_probs) 
                })
                # also print ROC info to stdout for local log file
                print(f"AUC: {auc:.4f}")
                print("FPR:", np.round(fpr, 4))   # round for cleaner logs
                print("TPR:", np.round(tpr, 4))
                print(f"ROC Curve Points: {len(fpr)} thresholds")

                # save checkpoint after each epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(BACKUP_DIR, f"checkpoint_{data_dir_label}_epoch_{epoch}.pt"))
            
                # be safe on GPU cache/memory issue
                gc.collect()                      # Collect garbage
                torch.cuda.empty_cache()         # Clear CUDA cache
            
                # -------- Early Stopping Logic --------
                if val_loss_avg < best_val_loss:  # check for improvement
                    best_val_loss = val_loss_avg
                    epochs_without_improvement = 0
                    print("[INFO] Validation loss improved.")
                else:
                    epochs_without_improvement += 1
                    print(f"[INFO] No improvement. Patience counter: {epochs_without_improvement}/{patience}")
                    if epochs_without_improvement >= patience:
                        print("[EARLY STOPPING] No improvement for consecutive epochs. Stopping early.")
                        break

            # compute training time
            total_time = time.time() - start_time
            print(f"Total training time: {total_time:.2f} seconds")

            # save model with timestamp and config info --> encoder-lang specific name
            model_filename = f"fc_model_{data_dir_label}_ctx{context_length}_ep{epoch+1}_bs{batch_size}_{timestamp}.pth"  # save with actual trained epoch
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved to {model_filename}")

            wandb.finish()

    return model  # return the trained model for later use if needed

########################
# TOCONFIG check media universe path
media = "/PATH/TO/ALL_MEDIA"
# TOCONFIG list of train dir
train_list = [
    "XXX",
    "XXX",
    "XXX"
]
# TOCONFIG train full path
for train_dir_name in train_list:
    train_full_path = "/BASE/PATH/"+train_dir_name
    print(f"======== Training on {train_dir_name} started at {time.ctime()} ========")
    classifier = train_model(data_dir=train_full_path, media_folder=media, context_length=3)
    print(f"======== Training on {train_dir_name} ended at {time.ctime()} ========")
