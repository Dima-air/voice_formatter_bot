import numpy as np

def split_audio_into_chunks(audio_array, sampling_rate, chunk_duration_sec=25.0, overlap_sec=1.0):
    chunk_size = int(chunk_duration_sec * sampling_rate)
    overlap_size = int(overlap_sec * sampling_rate)
    step = chunk_size - overlap_size
    chunks = []
    start = 0
    while start < len(audio_array):
        end = min(start + chunk_size, len(audio_array))
        chunk = audio_array[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        chunks.append(chunk)
        start += step
    return chunks