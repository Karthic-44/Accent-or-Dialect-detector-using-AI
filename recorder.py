import os 
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np 

from predict import predict
sr = 16000
duration = 5

def record(duration :int = duration, sr: int = sr) -> np.ndarray:
    print("recording")
    audio = sd.rec(int(sr*duration), samplerate=sr,channels=1,dtype="float32")
    sd.wait()
    print("recorded")
    return audio

def main():
    while True:
        try:
            input("Press enter to start recording")
        except KeyboardInterrupt:
            print("Exiting")
            break
        
        audio = record()
        audio_i16 = (audio*32767).astype("int16")
        
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tmp:
            tmp_path = tmp.name
            wav.write(tmp_path,sr,audio_i16)
        
        try:
            result = predict(tmp_path)
            print(f"Accent: {result['accent']}")
            print(f"confidence: {result['confidence']*100:.2f}")
        except Exception as e:
            print(e)
        finally:
            os.remove(tmp_path)
    
if __name__ == "__main__":
    main()
