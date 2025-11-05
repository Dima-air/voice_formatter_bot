from asr import WhisperASR
from nlu import NLUModel
from formatter import TextFormatter
from utils import split_audio_into_chunks

class VoiceToFormattedTextPipeline:
    def __init__(self, whisper_model_path, nlu_model_path):
        self.asr = WhisperASR(whisper_model_path)
        self.nlu = NLUModel(nlu_model_path)
        self.formatter = TextFormatter()

    def process(self, audio_array, sampling_rate=16000):
        full_text = []
        chunks = split_audio_into_chunks(audio_array, sampling_rate)
        for chunk in chunks:
            transcription = self.asr.transcribe(chunk, sampling_rate)
            full_text.append(transcription)
        raw_text = " ".join(full_text)
        intent, entity = self.nlu.predict(raw_text)
        formatted = self.formatter.apply_formatting(raw_text, entity, intent)
        return formatted, raw_text, intent, entity