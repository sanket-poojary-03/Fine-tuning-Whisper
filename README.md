# Fine tuning Open Source Whisper (Speech-to-Text) Model

For the **DARPG Hackathon**, **The Problem Statement 3** involved evaluating and optimizing the Whisper model to accurately transcribe feedback calls related to citizen grievances into English text.

Since the textual output data was not provided, Whisper LLM was used to generate textual data for each audio dataset. This data was then stored in a `metadata.csv` file, and after preprocessing, it was used to fine-tune the Whisper small LLM.

## Dataset Preparation

Prepare your Audio folder in the following format:
```
audio_dataset/
├── metadata.csv
└── data/
```
`metadata.csv` contains the names of the audio files `audio_path` and their corresponding texts `transcription`.
