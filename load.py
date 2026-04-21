import librosa
import torch
import numpy as np

target_sr = 16000
max_sample = target_sr * 5

def load(path : str):
    
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    audio = audio.astype(np.float32)
    
    if len(audio) < max_sample:
        audio = np.pad(audio, (0, max_sample -len(audio)), mode="constant")
    
    else:
        audio = audio[:max_sample]
    
    return torch.from_numpy(audio)

# for testing:
 
# if __name__ == "__main__":
#     print(load(r"F:\dataset\cv-corpus\en\clips\common_voice_en_1.mp3"))
