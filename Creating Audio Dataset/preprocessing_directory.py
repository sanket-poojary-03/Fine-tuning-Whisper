import pandas as pd
from transformers import AutoTokenizer
import os

'''
After splitting the audio dataset and generating the corresponding transcriptions.

The below function is used to check the sequence of the transcribed text and if the text exceeds the max_length,
deleting those particular files from the dataset
'''
def check_sequence_length(df, max_sequence_length=1024):
    for index, row in df.iterrows():
        text = row['transcription']
        tokens = tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) > max_sequence_length:
            print(f"Text at index {index} exceeds the max sequence length: {len(tokens)}")
            print(f"Text: {text}\n")

check_sequence_length(df)


audio_directory = 'location'
csv_files = set(df['file_name'])
all_audio_files = os.listdir(audio_directory)

files_to_remove = [file for file in all_audio_files if file not in csv_files]

for file_to_remove in files_to_remove:
    file_path = os.path.join(audio_directory, file_to_remove)
    os.remove(file_path)
