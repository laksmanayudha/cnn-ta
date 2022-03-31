import pandas as pd
import numpy as np
import time

def class_distribution(used_df):
    labels = set(used_df['category'])
    for label in labels:
        total_data = len(filter_dataframe(used_df, 'category', label))
        print(label, ":", total_data)
        
def filter_dataframe(df, column, name):
    return df[df[column] == name]

def unique_words(df):
    tokens = []
    for data in df:
        for word in data.split():
            if word not in tokens:
                tokens.append(word)
    return tokens

def select_sample(df, n):
    labels = set(df['category'])
    concat = [filter_dataframe(df, 'category', label).sample(n, random_state=1) for label in labels]
    return pd.concat(concat, ignore_index=True)

def save_to_csv(df, name):
    df.to_csv(name)

def minutes(time):
    print(time/60, "minutes")


def filter_data(df, column, values):
    return df[df[column].isin(values)]