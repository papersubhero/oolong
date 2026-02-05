import pandas as pd
import numpy as np
import warnings
import re
import random
from typing import List
from pathlib import Path
from datetime import datetime
import os
import time

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

def get_random_sticker(dir_path: str) -> str:
    """
    Select a random file from the given directory and return its name only.

    Args:
        dir_path (str): Path to the directory

    Returns:
        str: Filename of the randomly chosen file, or None if no files found
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a valid directory.")

    # os.scandir is faster than iterdir
    with os.scandir(dir_path) as it: 
        files = [entry.name for entry in it if entry.is_file()] # file name only

    if not files:
        return None

    return random.choice(files)

def get_random_link(csv_path: str) -> str:
    """
    Efficiently select a random string from a CSV file where each row contains one string using reservoir sampling.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        str: A randomly chosen string, or None if the file is empty or invalid
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            chosen = None
            count = 0
            for line in f:
                # remove the link break mark at the end of line
                # otherwise the url will be appended with a line break
                line = line.strip()
                if not line:
                    continue
                count += 1
                # Reservoir sampling
                if random.randint(1, count) == 1:
                    chosen = line
        return chosen
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def run_dict(
    df: pd.DataFrame,
    prob: List[float],
    RANDSEED: int,
    link_universe_path: str,
    media_universe_path: str,
    perb: int, 
    llm: int
) -> pd.DataFrame:
    """
    Processes a DataFrame into a stego_df with left/right sender logic and open/stego message structure.

    Args:
        df (pd.DataFrame): Input dataframe
        llm (int): Binary flag
        prob (List[float]): List of probabilities
        RANDSEED (int): Random seed
        link_universe_path (str): CSV path for link universe
        media_universe_path (str): Dir path for media universe
        perb (int): Binary flag

    Returns:
        pd.DataFrame: A new dataframe called stego_df
    """

    # determine left_uid and right_uid
    unique_senders = sorted(df['sender_id'].dropna().unique())
    if len(unique_senders) == 2:
        left_uid, right_uid = unique_senders
    elif len(unique_senders) == 1: # a small amount of conv only has one user
        left_uid = right_uid = unique_senders[0]
    else:
        raise ValueError("Expected 1 or 2 unique sender_id values in df")

    # split df into two halves
    midpoint = (len(df) + 1) // 2  # round up to include middle in first half
    stego_df = df.iloc[:midpoint].copy()
    open_msg_df = df.iloc[midpoint:].copy()

    # append rows until stego_df is as long as original
    while len(stego_df) < len(df):
        r_side = random.random()
        stego_uid = left_uid if r_side < 0.5 else right_uid # 50-50 prob which side sends stego

        r_stego = random.random()
        if r_stego < prob[0]:  # the first value is stego prob
            r_stego_type = random.random()
            # df format 
            # id = -1, double circum label
            # sender_id = stego_uid
            # msg = np.nan, since other element msg are mostly nan
            # timestamp = np.nan, we do not use ts
            # reply_to_msg_id = np.nan,
            # media_location
            # has_media = 1,
            # reactions = np.nan,
            # media_type <-- 
            # label = "circum"
            this_row = []
            if r_stego_type <= prob[1]:
                # Add a video-type row
                this_row = [-1, stego_uid, np.nan, np.nan, np.nan, np.nan, 1, np.nan, "video", "circum"]
            elif r_stego_type <= prob[1] + prob[2]:
                # Add an image-type row
                this_row = [-1, stego_uid, np.nan, np.nan, np.nan, np.nan, 1, np.nan, "image", "circum"]
            elif r_stego_type <= prob[1] + prob[2] + prob[3]:
                # Add a sticker with sticker path
                media_path = get_random_sticker(media_universe_path)
                this_row = [-1, stego_uid, np.nan, np.nan, np.nan, media_path, 1, np.nan, "sticker", "circum"]
                    
            elif r_stego_type <= prob[1] + prob[2] + prob[3] + prob[4]:
                # Add a link-type row
                chosen_link = get_random_link(link_universe_path)
                this_row = [-1, stego_uid, chosen_link, np.nan, np.nan, np.nan, 0, np.nan, "link", "circum"]
            else:
                # Add an 'other' type row
                this_row = [-1, stego_uid, np.nan, np.nan, np.nan, np.nan, 1, np.nan, "other", "circum"]
                
            
            stego_df.loc[len(stego_df)] = this_row
        else:
            # Pull a row from open_msg_df
            if not open_msg_df.empty:
                stego_df.loc[len(stego_df)] = open_msg_df.iloc[0]
                open_msg_df = open_msg_df.iloc[1:]
            else:
                break
    ###

    return stego_df

# generate train dataset using dictionary method
def generate_classifier_dataset(
    train_dataset_path: str,
    output_path: str,
    prob: List[float],
    RANDSEED: int = 42,
    link_universe_path: str = "",
    media_universe_path: str = "",
    perb: int = 0,
    llm: int = 0
):
    train_path = Path(train_dataset_path)
    output_base = Path(output_path)

    if not train_path.is_dir():
        raise ValueError(f"Invalid train_dataset_path: {train_dataset_path}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probstr = concat_first_decimal_digits(prob)
    output_dir_name = train_path.name + f"_dict_{timestamp}" + "_probstr" + probstr 
    full_output_dir = output_base / output_dir_name
    full_output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(train_path.glob("*.csv"))
    read_count = 0
    write_count = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            read_count += 1

            df_with_label = clean_media_type_add_label(df)
            
            cirum_train_df = run_dict(
                df_with_label,
                prob=prob,
                RANDSEED=RANDSEED,
                link_universe_path=link_universe_path,
                media_universe_path=media_universe_path,
                perb=perb, 
                llm=llm
            )

            output_file_path = full_output_dir / csv_file.name
            cirum_train_df.to_csv(output_file_path, index=False)
            write_count += 1
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    print(f"Origninal conversation CSV files read: {read_count}")
    print(f"Stegochat conversation CSV files written: {write_count}")

# generate test dataset using dictionary method
def generate_test_dataset(
    test_dataset_path: str,
    output_path: str,
    prob: List[float],
    RANDSEED: int = 42,
    link_universe_path: str = "",
    media_universe_path: str = "",
    perb: int = 0,
    llm: int = 0,
    circum_population_rate: float = 0.2
):
    test_path = Path(test_dataset_path)
    output_base = Path(output_path)

    if not test_path.is_dir():
        raise ValueError(f"Invalid test_dataset_path: {test_dataset_path}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probstr = concat_first_decimal_digits(prob)
    circum_popu_rate_str = str(circum_population_rate).split(".")[1][0]
    output_dir_name = test_path.name + f"_dict_{timestamp}" + "_probstr" + probstr + "_circumrate" + circum_popu_rate_str
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

            output_file_path = full_output_dir / csv_file.name
            r_circum = random.random() # not all population would circumvent 
            if r_circum <= circum_population_rate:
                cirum_df = run_dict(
                    df_with_label,
                    prob=prob,
                    RANDSEED=RANDSEED,
                    link_universe_path=link_universe_path,
                    media_universe_path=media_universe_path,
                    perb=perb, 
                    llm=llm
                )
                # write the circum csv to output folder
                cirum_df.to_csv(output_file_path, index=False)
            else:
                # not circumveting population, write the original cleaned-up csv to output folder
                df_with_label.to_csv(output_file_path, index=False)
            write_count += 1
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    print(f"Origninal conversation CSV files read: {read_count}")
    print(f"Stegochat conversation CSV files written: {write_count}")

################

#### TOCONFIG train parameter
train_dataset_path = "/PATH/TO/TRAIN/DATA/"
train_output_path = "/PATH/TO/OUTPUT/FOLDER/"
train_RANDSEED = 42 # in case of need to pin randomness for debugging
train_link_universe_path = "/PATH/TO/ALL_LINK.csv"
train_media_universe_path = "/PATH/TO/ALL/MEIDA/"
train_perb = 0  
train_llm = 0   

#### TOCONFIG test parameter
test_dataset_path = "/PATH/TO/TEST/DATA/"
test_output_path = "/PATH/TO/OUTPUT/FOLDER/"
test_RANDSEED = 42
test_link_universe_path = "/PATH/TO/ALL_LINK.csv"
test_media_universe_path = "/PATH/TO/ALL/MEDIA/"
test_perb = 0  
test_llm = 0   
test_circum_population_rate = X # circumventing population percentage parameter, e.g., 0.2

#### TOCONFIG list of stego ratio, e.g., prob_list = [1, 0.5, 0.2, 0.1, 0.05]
prob_list = [X, X, X]

for prob_value in prob_list:
    ## total prob, video, image, sticker, link, other
    # e.g., train_prob = [0.1, 0.1, 0.1, 0.4, 0.3, 0.1], test_prob = [0.1, 0.1, 0.1, 0.4, 0.3, 0.1]
    # we mainly use sticker and link to critically eval our design (since those are the two types which we use as dictionary items), and use a few of video, image, other to evaluate the metadata part together
    train_prob = [prob_value, X, X, X, X, X] 
    test_prob = [prob_value, X, X, X, X, X]

    print(f"[START] generate data with prob {str(prob_value)} at {time.ctime()}")
    # call train function with train parameters
    generate_classifier_dataset(
        train_dataset_path=train_dataset_path,
        output_path=train_output_path,
        prob=train_prob,
        RANDSEED=train_RANDSEED,
        link_universe_path=train_link_universe_path,
        media_universe_path=train_media_universe_path,
        perb=train_perb,
        llm=train_llm
    )
    print(f"[DONE] train data with prob {str(prob_value)} on {train_dataset_path} at {time.ctime()}")
    
    generate_test_dataset(
        test_dataset_path=test_dataset_path,
        output_path=test_output_path,
        prob=test_prob,
        RANDSEED=test_RANDSEED,
        link_universe_path=test_link_universe_path,
        media_universe_path=test_media_universe_path,
        perb=test_perb,
        llm=test_llm,
        circum_population_rate=test_circum_population_rate
    )
    print(f"[DONE] test data with prob {str(prob_value)} on {test_dataset_path} at {time.ctime()}")

