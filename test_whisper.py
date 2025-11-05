from asr import WhisperASR
import librosa

asr = WhisperASR("./models/whisper-small-ru-final")
audio, sr = librosa.load("test.wav", sr=16000)
text = asr.transcribe(audio, sr)
print("Распознано:", text)