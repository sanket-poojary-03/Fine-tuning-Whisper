from datasets import load_dataset
from datasets import Audio
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
dataset = load_dataset("audiofolder", data_dir="/content/drive/MyDrive/audio/data")



model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")



feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")



tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # resample audio data from 48 to 16kHz
    audio = batch["audio"]

    batch["input_length"] = len(batch["audio"])

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    batch["labels_length"] = len(batch["labels"])
    return batch


MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000


def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < max_input_length

max_label_length = model.config.max_length

def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < max_label_length

dataset = dataset.map(prepare_dataset, remove_columns= dataset.column_names["train"])

dataset = dataset.filter(filter_inputs, input_columns=["input_length"])

dataset = dataset.filter(filter_labels, input_columns=["labels_length"])
