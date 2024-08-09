import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

'''Importing the whisper large-v3 model to generate textual data for the corresponding audio folder'''

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device)

''''The below code takes chunks of 100 audio files from the audio folder, sends them to the transcription pipeline to generate corresponding text,
 and appends the results to the metadata.csv file'''

def transcribe_and_append(audio_files, output_csv):
    if os.path.exists(output_csv) and os.stat(output_csv).st_size > 0:
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=["audio_path", "transcription"])

    # Iterating
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        audio_path = os.path.join(audio_folder, audio_file)
        result = pipe(audio_path, generate_kwargs={"language": "english"})

        transcriptions = result["text"]
        transcription = " ".join(transcriptions)
        df = df.append({"audio_path": audio_path, "transcription": transcription}, ignore_index=True)


    df.to_csv(output_csv, index=False)
    print(f"Transcriptions saved to {output_csv}")


audio_folder = "audio folder location"
output_csv = "metadata.csv"
chunk_size = 100
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".mp3")]
for i in range(0, len(audio_files), chunk_size):
    chunk_files = audio_files[i:i+chunk_size]
    transcribe_and_append(chunk_files, output_csv)
