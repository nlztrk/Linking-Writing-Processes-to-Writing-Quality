# %%writefile utils.py

import warnings
import os
import gc
import re
import random
from collections import Counter
import string
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from textstat import *
import lightgbm as lgb
import optuna
from unidecode import unidecode
from itertools import product
import polars as pl


# ## Text Extraction

def revealing_text(df):
    USER_ID = df["id"].iloc[0]
    textInputDf = df[['activity', 'cursor_position', 'text_change']]
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']
    currTextInput = textInputDf[['activity', 'cursor_position', 'text_change']]
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] +\
            essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), 
                        int(valueArr[0][1][:-1]), 
                        int(valueArr[1][0][1:]), 
                        int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] +\
                    essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] +\
                    essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]

    return USER_ID, essayText


# ## Text Features

def get_text_chunk_features(df, full_text=True):
    # Length of the text
    df['text_length'] = df['revealed_text'].apply(len)

    # Number of newlines and tabs
    df['num_newlines'] = df['revealed_text'].apply(lambda x: x.count('\n'))
    df['automated_readability_index'] = df['revealed_text'].apply(automated_readability_index)
    df['mcalpine_eflaw'] = df['revealed_text'].apply(mcalpine_eflaw)        
    df['coleman_liau'] = df['revealed_text'].apply(coleman_liau_index)

    # Repetitiveness (ratio of 'q' to total characters)
    df['repetitiveness'] = df['revealed_text'].apply(lambda x: x.count('q') / max(len(x), 1))

    # Average Word Length
    df['avg_word_length'] = df['revealed_text'].apply(lambda x: sum(len(word) for word in x.split()) / max(1, len(x.split())))

    # Word Count
    df['word_count'] = df['revealed_text'].apply(lambda x: len(x.split()))
    df['word_lexical_diversity'] = df['revealed_text'].apply(lambda x: len(set(x.split())) / len(x.split()))
        
    # Use of Quotations (count of quotation marks)
    df['num_s_quotations'] = df['revealed_text'].apply(lambda x: x.count("'"))
    df['num_d_quotations'] = df['revealed_text'].apply(lambda x: x.count('"'))
    
    df['qm_count'] = df['revealed_text'].apply(lambda x: x.count('?'))
    df['excm_count'] = df['revealed_text'].apply(lambda x: x.count('!'))
    df['comma_count'] = df['revealed_text'].apply(lambda x: x.count(','))
    df['dot_count'] = df['revealed_text'].apply(lambda x: x.count('.'))
    
    df['num_prelist_count'] = df['revealed_text'].apply(lambda x: x.count(':')) +\
                            df['revealed_text'].apply(lambda x: x.count(";"))    
    
    df["space_n_dot_mistake"] = df['revealed_text'].apply(lambda x: len(re.findall(r'\s\.', x)))
    df["space_n_comma_mistake"] = df['revealed_text'].apply(lambda x: len(re.findall(r'\s\,', x)))

    df["comma_n_nonspace_mistake"] = df['revealed_text'].apply(lambda x: len(re.findall(r'\,\S', x)))
    df["dot_n_nonspace_mistake"] = df['revealed_text'].apply(lambda x: len(re.findall(r'\.\S', x)))
    
    df["total_punc_mistake"] = (
        df["space_n_dot_mistake"] +
        df["space_n_comma_mistake"] +
        df["comma_n_nonspace_mistake"] +
        df["dot_n_nonspace_mistake"]
    )
    
    df["punc_mistake_ratio"] = df["total_punc_mistake"] / (df['qm_count'] +
                                                           df['excm_count'] +
                                                           df['comma_count'] +
                                                           df['dot_count'])

    # Count the number of unique words in each text
    df['unique_word_count'] = df['revealed_text'].apply(lambda x: len(set(re.findall(r'\w+', x.lower()))))
    
    # Count the number of different punctuation marks
    df['punctuation_count'] = df['revealed_text'].apply(lambda x: sum(x.count(p) for p in string.punctuation)) 
    

    return df


# +
def word_feats(df):
    essay_df = df
    df['word'] = df['revealed_text'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]

    word_agg_df = df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    df['sent'] = df['revealed_text'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len!=0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), 
                             df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df


def parag_feats(df):
    df['paragraph'] = df['revealed_text'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    # Number of characters in paragraphs
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    df['paragraph_sent_count'] = df['paragraph'].apply(lambda x: len(re.split('\\.|\\?|\\!',x)))

    df = df[df.paragraph_len>2].reset_index(drop=True)
    
    paragraph_agg_df = pd.concat([
        df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), 
        df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS),
        df[['id','paragraph_sent_count']].groupby(['id']).agg(AGGREGATIONS)       
    ], axis=1) 
    
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count", "paragraph_sent_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def product_to_keys(logs, essays):
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]

def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]


# +
# Helper functions

def q2(x):
    return x.quantile(0.2)
def q8(x):
    return x.quantile(0.8)

AGGREGATIONS = ['count',
#                 'mean',
                'min',
                'max',
                'first',
                'last',
#                 q2,
                'median',
#                 q8,
                'sum',
                'std']

def standardize_text(txt):
    txt = re.sub(r'\t' , '', txt)
    txt = re.sub(r'\n {1,}' , '\n', txt)
    txt = re.sub(r' {1,}\n' , '\n', txt)
    txt = re.sub(r'\n{2,}' , '\n', txt)
    txt = re.sub(r' {2,}' , ' ', txt)
    txt = txt.strip()
    return txt


def TextProcessor(inp_df):

    for rowi in range(len(inp_df)):
        if inp_df.loc[rowi, "revealed_text"].replace(" ", "") == "":
            inp_df.loc[rowi, "revealed_text"] = "q"   
    
    inp_df["revealed_text"] = inp_df["revealed_text"].apply(lambda x: standardize_text(txt=x))
    
    # features for complete text
    print("creating complete features")
    inp_df = get_text_chunk_features(inp_df) 
    
    wf_df = word_feats(inp_df)
    sf_df = sent_feats(inp_df)
    pf_df = parag_feats(inp_df)
    
    # merging all features
    inp_df = inp_df.merge(wf_df, how="left", on="id")
    inp_df = inp_df.merge(sf_df, how="left", on="id")
    inp_df = inp_df.merge(pf_df, how="left", on="id")
    
    return inp_df


# -

# ## Raw Logger Features

# ### Polars RawLog Features

# +
num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count', 'event_id']
activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']


def count_by_values(df, colname, values):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for i, value in enumerate(values):
        tmp_df = df.group_by('id').agg(
            pl.col(colname).is_in([value]).sum().alias(f'{colname}_{i}_cnt')
        )
        fts  = fts.join(tmp_df, on='id', how='left') 
    return fts

def event_count_feats(df):
    temp = count_by_values(df, 'activity', activities)
    temp = temp.join(count_by_values(df, 'text_change', text_changes), on='id', how='left') 
    temp = temp.join(count_by_values(df, 'down_event', events), on='id', how='left') 
    temp = temp.to_pandas()
    return temp


# -

def word_stat_feats(df):
    temp = df.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
    temp = temp.group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
    temp = temp.with_columns(input_word_count = pl.col('text_change').list.lengths(),
                             input_word_length_mean = pl.col('text_change').apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_max = pl.col('text_change').apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_std = pl.col('text_change').apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_median = pl.col('text_change').apply(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
                             input_word_length_skew = pl.col('text_change').apply(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)))
    temp = temp.drop('text_change')
    temp = temp.to_pandas()
    return temp


# +
def num_colstat_feats(df):
    temp = df.group_by("id").agg(
                                 pl.sum('action_time').suffix('_sum'),
#                                  pl.mean(num_cols).suffix('_mean'),
                                 pl.std(num_cols).suffix('_std'),
                                 pl.median(num_cols).suffix('_median'),
                                 pl.min(num_cols).suffix('_min'),
                                 pl.max(num_cols).suffix('_max'),
                                )
    temp = temp.to_pandas()
    return temp

def cat_colstat_feats(df):
    temp  = df.group_by("id").agg(pl.n_unique(['activity', 'down_event', 'up_event', 'text_change']))
    temp = temp.to_pandas()
    return temp


# +
def pause_stat_aggregator(df, prefix="iw"):
    temp = df.group_by("id").agg(
           # Inter-Key Pause Aggregations
           pl.max('time_diff').alias(f"{prefix}_max_pause_time"),
           pl.median('time_diff').alias(f"{prefix}_median_pause_time"),
           pl.mean('time_diff').alias(f"{prefix}_mean_pause_time"),
           pl.min('time_diff').alias(f"{prefix}_min_pause_time"),
           pl.std('time_diff').alias(f"{prefix}_std_pause_time"),
           pl.sum('time_diff').alias(f"{prefix}_total_pause_time"),
           # Inter-Key Pause Counters
           pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') <= 1)).count().alias(f"{prefix}_pauses_half_sec"),
           pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') <= 2)).count().alias(f"{prefix}_pauses_1_sec"),
           pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') <= 3)).count().alias(f"{prefix}_pauses_2_sec"),
           pl.col('time_diff').filter(pl.col('time_diff') > 3).count().alias(f"{prefix}_pauses_3_sec")
    )
    return temp

def idle_time_feats(df):
    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))

    temp = temp.with_columns((pl.col("up_event") == "Space").alias("is_space"))
    temp = temp.with_columns((pl.col("up_event") == ".").alias("is_dot"))
    temp = temp.with_columns((pl.col("up_event") == "Enter").alias("is_enter"))

    temp = temp.with_columns(
        pl.col("is_space").cumsum().shift().backward_fill().over("id").alias("word_id"),
        pl.col("is_dot").cumsum().shift().backward_fill().over("id").alias("sentence_id"),    
        pl.col("is_enter").cumsum().shift().backward_fill().over("id").alias("paragraph_id"),    
    )

    temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))

    # All (in-word) pauses
    iw_df = pause_stat_aggregator(df=temp, prefix="iw")

    # Between-words pauses
    bww_df = temp.group_by("id", "word_id").agg(pl.col("time_diff").first())
    bww_df = pause_stat_aggregator(df=bww_df, prefix="bww")

    # Between-sentences pauses
    bws_df = temp.group_by("id", "sentence_id").agg(pl.col("time_diff").first())
    bws_df = pause_stat_aggregator(df=bws_df, prefix="bws")

    # Between-paragraphs pauses
    bwp_df = temp.group_by("id", "paragraph_id").agg(pl.col("time_diff").first())
    bwp_df = pause_stat_aggregator(df=bwp_df, prefix="bwp")

    return_df = (
        iw_df
        .join(bww_df, on="id", how="left")
        .join(bws_df, on="id", how="left")
        .join(bwp_df, on="id", how="left")
                )
    
    return_df = return_df.to_pandas()
    
    return return_df


# +
def burst_features(df, burst_type="p"):
    
    temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
    temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
    
    if burst_type == "p":
        temp = temp.with_columns(pl.col('activity').is_in(['Input']))
    elif burst_type == "r":
        temp = temp.with_columns(pl.col('activity').is_in(['Remove/Cut']))
        
#     temp = temp.with_columns(pl.col('time_diff')<2)
    temp = temp.with_columns((pl.col('action_time') / 1000).alias("action_time_s"))
    temp = temp.with_columns((pl.col('up_time') / 1000).alias("up_time_s"))
    temp = temp.with_columns(pl.when(pl.col("activity")).then(pl.col("activity").rle_id())\
                             .alias(f'{burst_type}_burst_group'))
    temp = temp.drop_nulls()

    temp = temp.group_by("id", f"{burst_type}_burst_group").agg(
                                    # Character count features
                                    pl.count('activity').alias(f'{burst_type}_burst_group_keypress_count'),
                                    # Time features
                                    pl.sum('action_time_s').alias(f'{burst_type}_burst_group_timespent'),
                                    pl.mean('action_time_s').alias(f'{burst_type}_burst_keypress_timespent_mean'),
                                    pl.std('action_time_s').alias(f'{burst_type}_burst_keypress_timespent_std'),
                                    # First-last burst timestamps   
                                    pl.min('up_time_s').alias(f'{burst_type}_burst_keypress_timestamp_first'),
                                    pl.max('up_time_s').alias(f'{burst_type}_burst_keypress_timestamp_last')
    )

    temp = temp.group_by("id").agg(
        # Character count features
                pl.sum(f'{burst_type}_burst_group_keypress_count').alias(f'{burst_type}_burst_keypress_count_sum'),
                pl.mean(f'{burst_type}_burst_group_keypress_count').alias(f'{burst_type}_burst_keypress_count_mean'),
                pl.std(f'{burst_type}_burst_group_keypress_count').alias(f'{burst_type}_burst_keypress_count_std'),
                pl.max(f'{burst_type}_burst_group_keypress_count').alias(f'{burst_type}_burst_keypress_count_max'),
        # Time features
                pl.sum(f'{burst_type}_burst_group_timespent').alias(f'{burst_type}_burst_timespent_sum'),
                pl.mean(f'{burst_type}_burst_group_timespent').alias(f'{burst_type}_burst_timespent_mean'),
                pl.std(f'{burst_type}_burst_group_timespent').alias(f'{burst_type}_burst_timespent_std'),
                pl.max(f'{burst_type}_burst_group_timespent').alias(f'{burst_type}_burst_timespent_max'),
                pl.mean(f'{burst_type}_burst_keypress_timespent_mean').alias(f'{burst_type}_burst_keypress_timespent_mean'),
                pl.mean(f'{burst_type}_burst_keypress_timespent_std').alias(f'{burst_type}_burst_keypress_timespent_std'),
        # First-last burst timestamps        
                pl.min(f'{burst_type}_burst_keypress_timestamp_first').alias(f'{burst_type}_burst_keypress_timestamp_first'),
                pl.max(f'{burst_type}_burst_keypress_timestamp_last').alias(f'{burst_type}_burst_keypress_timestamp_last')
    )
    
    temp = temp.to_pandas()
    return temp


# -

# ## Whole Raw Processor Class

def RawProcessor(raw_df):
    
    raw_df_pl = pl.from_pandas(raw_df)

    ###
    print("Creating kpps features...")                
    feat_df = get_keys_pressed_per_second(raw_df)

    ###
    print("Creating event-count features...")                
    feat_df = feat_df.merge(event_count_feats(df=raw_df_pl), how="left", on="id") 
    
    print("Creating numerical-categorical aggregation features...")                
    feat_df = feat_df.merge(num_colstat_feats(df=raw_df_pl), how="left", on="id")    
    feat_df = feat_df.merge(cat_colstat_feats(df=raw_df_pl), how="left", on="id")  
#     feat_df = feat_df.merge(word_stat_feats(df=raw_df_pl), how="left", on="id")  
    
    print("Creating pause features...")   
    feat_df = feat_df.merge(idle_time_feats(df=raw_df_pl), how="left", on="id") 
    
    print("Creating PR-Burst features...")  
    feat_df = feat_df.merge(burst_features(df=raw_df_pl, burst_type="p"), how="left", on="id")    
    feat_df = feat_df.merge(burst_features(df=raw_df_pl, burst_type="r"), how="left", on="id")    
    
    # Per-minute normalization features
    feat_df["p_bursts_timeratio"] = feat_df["p_burst_timespent_sum"] / (feat_df["up_time_max"] / 1000)
    feat_df["r_bursts_timeratio"] = feat_df["r_burst_timespent_sum"] / (feat_df["up_time_max"] / 1000)
    feat_df["action_timeratio"] = feat_df["action_time_sum"] / feat_df["up_time_max"]
    
    feat_df["pause_timeratio"] = feat_df["iw_total_pause_time"] / (feat_df["up_time_max"] / 1000)
    feat_df["pausecount_timeratio"] = feat_df["iw_pauses_2_sec"] / (feat_df["up_time_max"] / 1000)
    
    feat_df['word_time_ratio'] = feat_df['word_count_max'] / (feat_df["up_time_max"] / 1000)
    feat_df['word_event_ratio'] = feat_df['word_count_max'] / (feat_df["up_time_max"] / 1000)
    feat_df['event_time_ratio'] = feat_df['event_id_max']  / (feat_df["up_time_max"] / 1000)
    
    ###    
    return feat_df
