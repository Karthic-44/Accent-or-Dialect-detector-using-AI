# Accent Detector

Record a few seconds of speech and this tells you which English accent you have. It's a fine-tuned wav2vec2 model trained on Mozilla Common Voice, and it hits **97% validation accuracy**.

## What it does

You give it an audio clip (a file, or 5 seconds from your mic) and it gives back:

- the most likely accent
- how confident it is
- the top 3 guesses with their probabilities

```python
from predict import predict

predict("sample.wav")
# {'accent': 'british',
#  'confidence': 0.93,
#  'top3': [{'accent': 'british', 'confidence': 0.93}, ...]}
```

## How it works

1. **Load the audio.** Every clip is resampled to 16 kHz mono and padded or cut to exactly 5 seconds, so batches are all the same shape.
2. **Run it through wav2vec2-base.** This is Facebook's pretrained speech model. I froze most of it and only fine-tuned the last two transformer layers, which keeps training fast and stops it from forgetting what it already knows.
3. **Attention pooling.** Instead of averaging all the time frames together, a small network learns which frames matter most for accent and weights them accordingly.
4. **A small classifier on top.** Two hidden layers (512 and 256 units) with LayerNorm, GELU and dropout, then one output per accent.

## Data

I used the English portion of [Mozilla Common Voice](https://commonvoice.mozilla.org/), which has self-reported accent labels. They're messy, so `preprocess.py` does some cleanup first: it drops clips with no label, lowercases everything, merges a few long names (`united states english` → `usa`, `england english` → `british`, and the India/Pakistan/Sri Lanka group → `south asia`), and throws out any accent with fewer than 100 clips.

## Results

| | |
|---|---|
| Validation accuracy | **97%** |
| Inference latency | **~100-200 ms** per 5-second clip |

## Running it

```bash
pip install torch transformers librosa pandas numpy scikit-learn scipy sounddevice
```

You'll also need `ffmpeg` installed so librosa can read Common Voice's `.mp3` files, and a GPU if you don't want training to take forever.

The dataset paths are hardcoded at the top of `preprocess.py` and `train.py`, so change those to wherever you put Common Voice. Then:

```bash
python preprocess.py   # clean the labels
python train.py        # fine-tune, saves model/model.pth
python recorder.py     # press Enter, talk for 5 seconds, get a prediction
```

## Files

- `preprocess.py` cleans the Common Voice labels
- `load.py` loads and standardises one audio file
- `model.py` wav2vec2, attention pooling and the classifier
- `train.py` the training and validation loop
- `predict.py` loads the checkpoint and runs inference
- `recorder.py` records from your mic and predicts
