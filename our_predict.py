from functools import lru_cache
from pathlib import Path
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from lipnet.model2 import LipNet
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from preprocess import extract

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "evaluation" / "models" / "overlapped-weights368.h5"
DICTIONARY_PATH = BASE_DIR / "common" / "dictionaries" / "grid.txt"
IMG_C, IMG_W, IMG_H = 3, 100, 50
FRAMES_N = 75
ABSOLUTE_MAX_STRING_LEN = 32


@lru_cache(maxsize=1)
def get_predictor():
    lipnet = LipNet(
        img_c=IMG_C,
        img_w=IMG_W,
        img_h=IMG_H,
        frames_n=FRAMES_N,
        absolute_max_string_len=ABSOLUTE_MAX_STRING_LEN,
        output_size=28,
    )

    adam = Adam(learning_rate=0.0001)
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f'Weights file not found at {WEIGHTS_PATH}')

    lipnet.model.load_weights(str(WEIGHTS_PATH))

    spell = Spell(path=str(DICTIONARY_PATH))
    decoder = Decoder(
        greedy=True,
        beam_width=200,
        postprocessors=[labels_to_text, spell.sentence],
    )

    return lipnet, decoder

def predict_from_video(video_path):
    print(f"Processing: {video_path}")

    frames = extract(video_path)

    if len(frames) == 0:
        print("No frames extracted")
        return ""

    frames = frames.astype(np.float32) / 255.0
    frames = np.stack([frames, frames, frames], axis=-1)
    frames = np.transpose(frames, (0, 2, 1, 3))
    if len(frames) < FRAMES_N:
        pad = np.zeros(
            (FRAMES_N - len(frames), IMG_W, IMG_H, IMG_C),
            dtype=np.float32
        )
        frames = np.concatenate([frames, pad], axis=0)
    else:
        frames = frames[:FRAMES_N]

    lipnet, decoder = get_predictor()
    frames = np.expand_dims(frames, axis=0)
    y_pred = lipnet.predict(frames)
    input_length = np.array([FRAMES_N])
    result = decoder.decode(y_pred, input_length)[0]

    print("Prediction:", result)

    return result.strip()