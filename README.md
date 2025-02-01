# Fine tuning Open Source Whisper (Speech-to-Text) Model

For the **DARPG Hackathon 2024**, **The Problem Statement 3** involved evaluating and optimizing an Open Source speech-to-text model to accurately transcribe feedback calls related to citizen grievances into English text.

Since the textual output data was not provided, Whisper LLM was used to generate textual data for each audio dataset. This data was then stored in a `metadata.csv` file, and after preprocessing, it was used to fine-tune the Whisper small LLM.

Here’s a YouTube video explaining our project: [Watch our project explanation on YouTube](https://youtu.be/qPTS3mdLkAY?si=xgYwI-QeYI0aC2Km) 


## Dataset Preparation

Prepare your Audio folder in the following format:
```
audio_dataset/
├── metadata.csv
└── data/
```
`metadata.csv` contains the names of the audio files `audio_path` and their corresponding texts `transcription`.
`data/` folder contains all the audio files.

## Hackathon Workflow  

These are all the steps we followed during the hackathon.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8aad9a03-1a5b-4214-8660-2c0f0aeb5021" alt="Hackathon Workflow">
</p>



## Deployment  

We have deployed this model on **Hugging Face Spaces** for easy access and usage. You can try it out here:  

🤗 [WHISPER-SPEECH-TO-TEXT-MODEL-FOR-DARPG](https://huggingface.co/spaces/sanket003/WHISPER-SPEECH-TO-TEXT-MODEL-FOR-DARPG)



## Using the Model Locally:

To use the model, run the `run_model.py` script, which contains a Gradio interface for easy interaction with the model.



