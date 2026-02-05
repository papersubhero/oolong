import pandas as pd
import numpy as np
import warnings
import re
import random
from typing import List
from pathlib import Path
from datetime import datetime
import os
import gc

# for better naming of the output dir name with prob
def concat_first_decimal_digits(floats: list[float]) -> str:
    result = ""
    for value in floats:
        s = str(value)
        if '.' in s:
            decimal_part = s.split('.')[1]
            digit = decimal_part[0] if decimal_part else '05' # first digit is 0 means we are at 0.05 experiment
        else: # no decimal, so we are at 1 experiment
            digit = '10'
        result += digit
    return result

# standardize media_type values
def map_media_type(row):
    mt = str(row['media_type']) if pd.notna(row['media_type']) else None
    msg = str(row['msg']) if pd.notna(row['msg']) else ""
    
    if mt in ["application/x-tgsticker", "image/webp"]:
        return "sticker"
    elif mt and ("image" in mt or "img" in mt):
        return "image"
    elif mt and "video" in mt:
        return "video"
    elif mt:
        return "other"
    elif not mt and "https" in msg:
        # URL-safe character check
        if re.fullmatch(r"https://[^\s]*", msg.strip()):
            return "link"
        else:
            return "none"
    else:
        return "none"

def clean_media_type_add_label(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist
    required_cols = ['sender_id', 'msg', 'media_type', 'media_location']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    df['media_type'] = df.apply(map_media_type, axis=1)

    # validate that media_type is in allowed set
    allowed_types = {"none", "image", "video", "sticker", "link", "other"}
    unique_types = set(df['media_type'].unique())
    unexpected = unique_types - allowed_types
    if unexpected:
        raise ValueError(f"Unexpected media_type values found: {unexpected}")

    # add default label column
    df['label'] = "noncircum"

    # ensure all required columns are present
    final_cols = ['sender_id', 'msg', 'media_type', 'media_location', 'label']
    for col in final_cols:
        if col not in df.columns:
            warnings.warn(f"Expected column '{col}' not found in DataFrame.")

    return df

# import libraries for discop
import sys
# TOCONFIG path to Discop/src
sys.path.append("/PATH/TO/Discop/src") 
# makes sure Discop/src/config.py can be imported in stega_cy
# print(sys.path)
# only import text part
from Discop.src.config import Settings, text_default_settings
from Discop.src.stega_cy import encode_text, decode_text, decode
from Discop.src.model import get_model, get_feature_extractor, get_tokenizer
# to avoid a deprecation error "`TransfoXL` was deprecated due to security issues linked to `pickle.load` in `TransfoXLTokenizer`"
os.environ["TRUST_REMOTE_CODE"] = "true" 


def generate_context(df_before, l_uid, r_uid, max_len=50):
    """
    Generate a dialogue context string from a dataframe of messages.
    
    Parameters:
        df_before (pd.DataFrame): Input dataframe containing at least 'sender_id' and 'msg'.
        l_uid (int or str): User ID to be labeled as User1.
        r_uid (int or str): User ID to be labeled as User2.
        max_len (int): max length of context; placeholder for future truncation.
        
    Returns:
        str: Formatted context string.
    """
    context = ""
    # Truncate to the last `max_len` rows if necessary
    if len(df_before) > max_len:
        df_before = df_before.tail(max_len)  # Keep last max_len rows

    for _, row in df_before.iterrows():
        sender = row.get('sender_id')
        msg = row.get('msg', "")
        if pd.isna(msg):
            msg = ""

        if sender == l_uid:
            context += "User1:\n" + msg
        elif sender == r_uid:
            context += "User2:\n" + msg
        else:
            continue  # Skip rows with unknown sender_id

        context += "\n"

    return context

# load pretrained discop model and tokenizer
import torch
settings: Settings = text_default_settings
settings.device = torch.device('cuda:0')
settings.model_name = "transfo-xl-wt103"
discop_model = get_model(settings)
discop_tokenizer = get_tokenizer(settings)

def get_stego_discop(input_len_list: List[int], context: str):    
    # TOCONFIG generate X bits ciphertext for en dataset, can be adapted to different capacity by referring to input_len_list
    input_str = ''.join(random.choice(['0', '1']) for _ in range(X))
    # call the encode function with the max length output only once
    # TOCONFIG settings.length = max_output_len, e.g., 5 for 15 bits
    settings.length = X
    try:
        output = encode_text(discop_model, discop_tokenizer, input_str, context, settings)
        # get the token ids to do tuncation into different output length 
        output_id_list = output.generated_ids
        # get the context ids for decoding to bitstring
        context_ids = discop_tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(settings.device)
        # if the following loop does not find the exact secret string, just use all the output tokens
        exact_output = output
        for l in range(1,6):
            stego_ids = output_id_list[:l]
            bitstring_decoded = decode(discop_model, context_ids, stego_ids, settings)
            if input_str in bitstring_decoded:
                # use the exact number of tokens
                exact_output = discop_tokenizer.decode(output_id_list[:l])
                # break once matched
                break

        # remove reference to output once it's used 
        del output

    except Exception as e:
        # gracefully handle generation failure
        print(f"[Discop Failure] Failed with context == {context} == {e}")
        exact_output = ""

    # cleanup after each generation
    gc.collect()
    torch.cuda.empty_cache()

    return exact_output, input_str # return stego message and ciphertext for later decoding correctness testing

def run_discop_with_ct(
    df: pd.DataFrame,
    prob: List[float],
    input_len_list: List[int]
):
    """
    Processes a DataFrame into a stego_df with left/right sender logic and open/stego message structure.

    Args:
        df (pd.DataFrame): Input dataframe, preprocessed chat history
        prob (List[float]): List of probabilities, for discop, only one prob is enough for stego ratio
        input_len_list: six different input length for generating stego text

    Returns:
        pd.DataFrame: new dataframe called stego_df
    """

    # determine left_uid and right_uid
    unique_senders = sorted(df['sender_id'].dropna().unique())
    if len(unique_senders) == 2:
        left_uid, right_uid = unique_senders
    elif len(unique_senders) == 1: # a small amount of conv only has one user
        left_uid = right_uid = unique_senders[0]
    else:
        raise ValueError("Expected 1 or 2 unique sender_id values in df")

    # append a column to record ciphertext for stego rows
    df['ct'] = "no ciphertext"

    # split df into two halves
    midpoint = (len(df) + 1) // 2  # round up to include middle in first half
    stego_df = df.iloc[:midpoint].copy()
    open_msg_df = df.iloc[midpoint:].copy()
    
    # append rows until stego_df is as long as original
    while len(stego_df) < len(df): 
        r_side = random.random()
        stego_uid = left_uid if r_side < 0.5 else right_uid # 50-50 prob which side sends stego

        r_stego = random.random()
        if r_stego < prob[0]:  # prob = [0.2] the first value is stego prob
            # df format 
            # id = -1, not used
            # sender_id = stego_uid
            # msg = stego
            # timestamp = np.nan, not used
            # reply_to_msg_id, not used
            # media_location, nan
            # has_media = 0, not used
            # reactions, not used
            # media_type = "none"
            # label = "circum"
            context_str = generate_context(stego_df, left_uid, right_uid)
            # return a list of stego string with different length input and output
            stego_txt, ciphertext = get_stego_discop(input_len_list, context_str)

            # append the stego output df
            row_output = [-1, stego_uid, str(stego_txt), np.nan, np.nan, np.nan, 0, np.nan, "none", "circum", ciphertext]
            stego_df.loc[len(stego_df)] = row_output
            
        else:
            # pull a row from open_msg_df
            if not open_msg_df.empty:
                # append the next open msg stego df
                stego_df.loc[len(stego_df)] = open_msg_df.iloc[0]
                
                # pop the just used open_msg from the original open msg df
                open_msg_df = open_msg_df.iloc[1:]
            else:
                break

    return stego_df

def generate_discop_classifier_dataset(
    train_dataset_path: str,
    output_path: str,
    prob: List[float]
):
    train_path = Path(train_dataset_path)
    output_base = Path(output_path)

    if not train_path.is_dir():
        raise ValueError(f"Invalid train_dataset_path: {train_dataset_path}")

    # create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probstr = concat_first_decimal_digits(prob)
    
    # one input dataset results for six output dataset
    # TOCONFIG add Discop model into the output folder name
    output_dir_name = train_path.name + f"_discop_MODELNAME_{timestamp}" + "_probstr" + probstr 
    full_output_dir = output_base / output_dir_name 
    full_output_dir.mkdir(parents=True, exist_ok=True) 
    
    csv_files = list(train_path.glob("*.csv"))
    read_count = 0
    write_count = 0

    # DEBUG [:3] 
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            read_count += 1

            df_with_label = clean_media_type_add_label(df)
            
            cirum_train_df = run_discop_with_ct(df_with_label,prob=prob)
            output_file_path = full_output_dir / csv_file.name 
            cirum_train_df.to_csv(output_file_path, index=False) 
            
            write_count += 1
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    print(f"Origninal conversation CSV files read: {read_count}")
    print(f"Discop stegochat conversation CSV files written: {write_count}")

def generate_discop_test_dataset(
    test_dataset_path: str,
    output_path: str,
    prob: List[float],
    circum_population_rate: float 
):
    test_path = Path(test_dataset_path)
    output_base = Path(output_path)

    if not test_path.is_dir():
        raise ValueError(f"Invalid test_dataset_path: {test_dataset_path}")

    # prepare output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probstr = concat_first_decimal_digits(prob)
    circum_popu_rate_str = str(circum_population_rate).split(".")[1][0]

    # generate output dir
    # TOCONFIG add Discop model into the output folder name
    output_dir_name = test_path.name + f"_discop_MODELNAME_{timestamp}" + "_probstr" + probstr + "_circumrate" + circum_popu_rate_str
    full_output_dir = output_base / output_dir_name
    full_output_dir.mkdir(parents=True, exist_ok=True)
    

    csv_files = list(test_path.glob("*.csv"))
    read_count = 0
    write_count = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            read_count += 1

            df_with_label = clean_media_type_add_label(df)

            
            # write to output dir
            output_file_path = full_output_dir / csv_file.name 
            
            
            r_circum = random.random() # not all population would circumvent 
            if r_circum <= circum_population_rate:
                
                # take six different output df
                cirum_test_df = run_discop_with_ct(df_with_label,prob=prob)

                # write six test output csv
                cirum_test_df.to_csv(output_file_path, index=False) 
                
                
            else:
                # not circumveting population, write the original cleaned-up csv to all six output folder
                df_with_label.to_csv(output_file_path, index=False) 
            
            write_count += 1
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    print(f"Origninal conversation CSV files read: {read_count}")
    print(f"Discop stegochat conversation CSV files written: {write_count}")

#### TOCONFIG train-test parameters
train_dataset_path = "/PATH/TO/TRAIN/DATA/"
train_output_path = "/PATH/TO/OUPUT/" 

test_dataset_path = "/PATH/TO/TEST/DATA/"
test_output_path = "/PATH/TO/OUPUT/"

#### TOCONFIG prob, e.g., prob_list = [[1], [0.5], [0.2], [0.1], [0.05]]
prob_list = [[X], [X]]

import time

for prob_value_as_list in prob_list:
    # TOCONFIG change the print info to indicate the corresponding process
    print(f"[START] generate discop data with prob {str(prob_value_as_list)} at {time.ctime()}")
    generate_discop_classifier_dataset(
        train_dataset_path=train_dataset_path,
        output_path=train_output_path,
        prob=prob_value_as_list
    )
    # TOCONFIG change the print info to indicate the corresponding process
    print(f"[DONE] discop train data with prob {str(prob_value_as_list)} on {train_dataset_path} at {time.ctime()}")
    generate_discop_test_dataset(
        test_dataset_path=test_dataset_path,
        output_path=test_output_path,
        prob=prob_value_as_list
    )
    # TOCONFIG change the print info to indicate the corresponding process
    print(f"[DONE] discop test data with prob {str(prob_value_as_list)} on {test_dataset_path} at {time.ctime()}")


