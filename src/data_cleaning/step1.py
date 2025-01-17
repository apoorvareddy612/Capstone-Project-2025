#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:04:24 2025

@author: apoorvareddy
"""
import pandas as pd 
import numpy as np 

df = pd.read_csv('podcastdata_dataset.csv')

#Getting insights about dataset
print(df['title'].nunique())
print(df['text'].nunique())

#Data Cleaning
df = df[['title','text']]
print(df[df.duplicated(subset=['title'], keep='first')])
#remove the duplicates
mask = df.duplicated(subset=['title'], keep='last')
df = df[~mask]
#cehck for null values and remove it 
print(df.isna().sum())

df.to_csv('lex_fridman.csv',index=False)

df1 = pd.read_csv('skeptoid_transcripts.csv')
print(df1.head())

#Getting insights about dataset
print(df1['title'].nunique())
print(df1['text'].nunique())

#Data Cleaning
df1 = df1[['title','text']]
#cehck for null values and remove it 
print(df1.isna().sum())
df1 = df1.dropna()
#check for duplicates
print(df1[df1.duplicated(subset=['title'],keep=False)])
#concatenate values 
df1 = df1.groupby('title')['text'].agg(' '.join).reset_index()

df1.to_csv('skeptoid.csv',index=False)

df2 = pd.read_csv('VOX_today_explained_podcast_transcripts.csv')
print(df2.head())

#Getting insights about dataset
print(df2['episodeName'].nunique())
print(df2['text'].nunique())

#Data Cleaning
df2 = df2[['episodeName','text']]
#cehck for null values and remove it 
print(df2.isna().sum())
df2 = df2.dropna()
#check for duplicates
print(df2[df2.duplicated(subset=['episodeName'],keep=False)])
#concatenate values 
df2 = df2.groupby('episodeName')['text'].agg(' '.join).reset_index()

df2.to_csv('vox.csv',index=False)


import os
import json
import pandas as pd

# Set paths for the data folder and files
data_folder = 'HubermanLabTranscripts'  # Replace with your actual folder path
text_folder = os.path.join(data_folder, 'text')  # Folder containing .txt files
json_file = os.path.join(data_folder, 'videoID.json')  # JSON file with videoId and title mapping

# Load videoId to title mapping from the JSON file
with open(json_file, 'r') as f:
    video_titles = json.load(f)

# Prepare a list to hold the data (title, text)
data = []

# Iterate over all .txt files in the 'text' folder
for filename in os.listdir(text_folder):
    if filename.endswith('.txt'):
        video_id = filename.split('.')[0]  # Extract videoId from the filename (e.g., '123.txt' -> '123')
        text_path = os.path.join(text_folder, filename)
        
        # Read the content of the text file
        with open(text_path, 'r', encoding='utf-8') as text_file:
            text_content = text_file.read()
        
        # Get the title for this videoId from the video_titles dictionary
        title = video_titles.get(video_id, 'Unknown Title')  # Default to 'Unknown Title' if not found
        
        # Add title and text to the data list
        data.append({'title': title, 'text': text_content})

# Convert the list of data to a DataFrame
df3 = pd.DataFrame(data)

df3.to_csv('huberman.csv',index=False)
























