"""
filter audio using an aligner via the coverage ratio technique

takes in a raw manifest file and then runs the audio stated in the manifest path to the aligner
outputs audio that aligns with the audio with a set threshold
"""

import os
import sys
import gc
import logging
import json
from tqdm import tqdm

import torch
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# Setup logging in a nice readable format
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s', datefmt='%H:%M:%S'
)

# to load the modules packages that is one level up from this code
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add to sys.path
sys.path.append(parent_dir)

from modules import load_manifest_nemo

class Aligner:

    def __init__(
        self,
        root_dir: str,
        input_manifest_dir: str,
        output_manifest_dir: str,
        model_dir: str,
        emission_batch_size: int,
        coverage_threshold: float,
    ) -> None:
        
        self.root_dir = root_dir
        self.input_manifest_dir = input_manifest_dir
        self.output_manifest_dir = output_manifest_dir
        self.emission_batch_size = emission_batch_size
        self.coverage_threshold = coverage_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.getLogger(__name__).info(f"Device: {self.device}")

        # === Load model and audio ===
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device,
            model_path=model_dir,
            dtype=torch.float32
        )

        # TODO: to support more languages in the future 
        self.iso_639_mapping = {
            "ar": "ara",
            "bn": "ben",
            "hi": "hin",
            "id": "ind",
            "ms": "msa",
            "th": "tha",
            "tl": "tgl",
            "vi": "vie",
            "en": "eng",
            "zh": "zho",
        }

        self.sample_rate = 16000

    def align(self) -> None:

        items = load_manifest_nemo(input_manifest_path=self.input_manifest_dir)
        items_new = []

        with open(self.output_manifest_dir, 'w', encoding='utf-8', buffering=1) as f_out:
            for idx, item in enumerate(tqdm(items)):

                audio_filepath = os.path.join(self.root_dir, item['audio_filepath'])
                try:
                    audio_waveform = load_audio(audio_filepath, self.alignment_model.dtype, self.alignment_model.device)
                
                    # generate alignments
                    emissions, stride = generate_emissions(self.alignment_model, audio_waveform, batch_size=self.emission_batch_size)
                    
                    # makeshift for chinese data due to the aligning issue if zh characters are not spaced
                    if item['language'] == "zh":
                        text = " ".join(list(item["text"]))
                    else:
                        text = item["text"]

                    tokens_starred, text_starred = preprocess_text(text, romanize=True, language=self.iso_639_mapping[item['language']])

                    segments, scores, blank_token = get_alignments(emissions, tokens_starred, self.alignment_tokenizer)
                    spans = get_spans(tokens_starred, segments, blank_token)
                    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

                    # compute duration coverage ratio
                    duration_audio = audio_waveform.shape[-1] / self.sample_rate
                    duration_transcription = sum(w['end'] - w['start'] for w in word_timestamps)
                    coverage_ratio = duration_transcription / duration_audio

                    logging.getLogger(__name__).info(f"Audio filepath: {item['audio_filepath']}")
                    logging.getLogger(__name__).info(f"Duration of audio: {duration_audio:.2f}s")
                    logging.getLogger(__name__).info(f"Aligned transcription duration: {duration_transcription:.2f}s")
                    logging.getLogger(__name__).info(f"Coverage ratio: {coverage_ratio:.2f}")

                    if coverage_ratio < self.coverage_threshold:
                        logging.getLogger(__name__).info(f"Skipping low coverage audio: {item['audio_filepath']}")
                        continue

                    torch.cuda.empty_cache()
                    gc.collect()

                except:
                    logging.getLogger(__name__).warning(f"Failed to decode audio {audio_filepath}, skipping sample...")
                    continue

                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

        # read the output manifest
        items_new = load_manifest_nemo(input_manifest_path=self.output_manifest_dir)

        duration_items = sum([x['duration'] for x in items])
        duration_items_new = sum([x['duration'] for x in items_new])

        logging.getLogger(__name__).info(f'Available number of hours BEFORE processing: {duration_items/3600:.5}h')
        logging.getLogger(__name__).info(f'Available number of hours AFTER processing: {duration_items_new/3600:.5}h')
        logging.getLogger(__name__).info(f"Total number of audio files: {len(items)}")
        logging.getLogger(__name__).info(f"Total number of audio files aligned: {len(items_new)}")
        logging.getLogger(__name__).info(f"Number of erroneous audio: {len(items)-len(items_new)}")


    def __call__(self) -> None:

        return self.align()
    
if __name__ == "__main__":

    SPLIT = 'train'
    SUBSET = f'yodas2/yodas2_id_{SPLIT}'
    ROOT_DIR = os.path.join('/datasets/open_source_data', SUBSET)
    # ROOT_DIR = os.path.join('/nas/processed', SUBSET)
    INPUT_MANIFEST_DIR = os.path.join(ROOT_DIR, f"{SPLIT}_manifest_3_15_250h_oversampled_esc_filtered_lid_filtered_1.json")
    OUTPUT_MANIFEST_DIR = os.path.join(ROOT_DIR, f"{SPLIT}_manifest_3_15_250h_oversampled_esc_filtered_lid_filtered_aligned_1.json")
    MODEL_DIR = "/models/mms-300m-1130-forced-aligner"
    EMISSION_BATCH_SIZE = 128
    COVERAGE_THRESHOLD = 0.5

    a = Aligner(
        root_dir=ROOT_DIR,
        input_manifest_dir=INPUT_MANIFEST_DIR,
        output_manifest_dir=OUTPUT_MANIFEST_DIR,
        model_dir=MODEL_DIR,
        emission_batch_size=EMISSION_BATCH_SIZE,
        coverage_threshold=COVERAGE_THRESHOLD
    )()