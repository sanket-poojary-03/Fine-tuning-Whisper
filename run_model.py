import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
import time

model_id = "sanket003/whisper-darpg"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True
)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=8,
    torch_dtype=torch.float32,
)

def transcribe_audio(audio, file):
    if audio:
        result = pipe(audio)
    elif file:
        result = pipe(file)
        pass
    else:
        result = {"text": "No input provided."}
    return result["text"]

iface = gr.Interface(
    title="Transforming Speech into Text",
    fn=transcribe_audio,
    inputs=[
        gr.Audio(sources="microphone", type="filepath", label="Record from Microphone"),
        gr.File(type="filepath", label="Upload Audio File"),
    ],
    outputs=["textbox"],
    description="Choose either microphone input or upload an audio file.",
)
iface.launch(share=True)
