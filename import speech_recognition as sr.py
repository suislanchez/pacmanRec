import speech_recognition as sr
from pydub import AudioSegment

# Convert mp3 to wav
audio_file_path = 'The AI Revolution is Rotten to the Core.mp3'
wav_file_path = 'The AI Revolution is Rotten to the Core.wav'
audio = AudioSegment.from_mp3(audio_file_path)
audio.export(wav_file_path, format='wav')

# Initialize recognizer
r = sr.Recognizer()

# Load the audio file
with sr.AudioFile(wav_file_path) as source:
    audio_data = r.record(source)
    try:
        # Recognize the speech in the audio file
        text = r.recognize_google(audio_data)
        print(text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")