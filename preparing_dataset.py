import os
import shutil
import pandas as pd
import random

'''The dataset is created by using the {CREATING AUDIO DATASET FOLDER}
    The function below divides the directory into subfolders for training and testing'''

data_folder = "audio/data"

train_folder = os.path.join(data_folder, "train")
test_folder = os.path.join(data_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

audio_files = [f for f in os.listdir(data_folder) if f.endswith(".mp3")]

split_percentage = 0.8

num_train_files = int(len(audio_files) * split_percentage)
num_test_files = len(audio_files) - num_train_files

random.shuffle(audio_files)

for i, audio_file in enumerate(audio_files):
    source_path = os.path.join(data_folder, audio_file)
    if i < num_train_files:
        destination_path = os.path.join(train_folder, audio_file)
    else:
        destination_path = os.path.join(test_folder, audio_file)

    shutil.move(source_path, destination_path)
print(f"{num_train_files} files moved to 'train' folder.")
print(f"{num_test_files} files moved to 'test' folder.")


'''The below function updates the {metadata.csv} file as we added two subfolders , train and test'''


data_folder = "audio/data"

train_folder = os.path.join(data_folder, "train")
test_folder = os.path.join(data_folder, "test")

csv_file_path = "audio/data/metadata.csv"
df = pd.read_csv(csv_file_path)


def update_file_name(file_name, folder):
    return os.path.join(folder, os.path.basename(file_name))

for file_name in os.listdir(train_folder):
    if file_name.endswith(".mp3"):
        df.loc[df['file_name'].str.endswith(file_name), 'file_name'] = update_file_name(file_name, 'train')
for file_name in os.listdir(test_folder):
    if file_name.endswith(".mp3"):
        df.loc[df['file_name'].str.endswith(file_name), 'file_name'] = update_file_name(file_name, 'test')

df.to_csv(csv_file_path, index=False)
print(f"CSV file updated with file paths based on 'train' and 'test' folders.")