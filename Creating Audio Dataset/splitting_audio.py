from pydub import AudioSegment
import os

'''This function creates data(by splitting the audio file) with audio length of approx ~ 30s as 
 Whisper feature extractor pads/truncates a batch of audio samples such that all samples have an input length of 30s '''
def split_audio(audio_file, output_folder, file_index):
    try:
        sound = AudioSegment.from_file(audio_file, format="mp3")
        segment_duration = 30 * 1000
        total_duration = len(sound)
        os.makedirs(output_folder, exist_ok=True)

        # Split the audio into segments of 30 seconds each
        for i, start in enumerate(range(0, total_duration, segment_duration)):
            end = min(start + segment_duration, total_duration)
            segment = sound[start:end]
            subfile_name = f"{file_index}-{i+1}.mp3"
            segment.export(os.path.join(output_folder, subfile_name), format="mp3", codec="libmp3lame")
            
        print(f"Audio file {audio_file} split into 30-second segments successfully!")

    except Exception as e:
        print(f"Error splitting {audio_file}: {e}")
        pass

audio_folder = "source location"
output_folder = "destination location"

audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith(".mp3")]

for index, audio_file in enumerate(audio_files, start=1):
    split_audio(audio_file, output_folder, index)
