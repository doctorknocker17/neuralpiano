import soundfile as sf
from pathlib import Path
import json
import note_seq
import multiprocessing
from tqdm import tqdm
from common import Base

def _process_track(track_files):
    """Helper function for multiprocessing to process a single track."""
    wav_file, midi_file = track_files
    try:
        info = sf.info(wav_file)
        ns = note_seq.midi_file_to_note_sequence(str(midi_file))
        ns = note_seq.apply_sustain_control_changes(ns)
        return (wav_file, ns, info.samplerate, info.frames)
    except Exception:
        return None

class Maestro(Base):
    def __init__(self, path: str, split: str = "train", **kwargs):
        path = Path(path)
        meta_file = path / "maestro-v3.0.0.json"
        
        # Normalize split name
        split = "validation" if split in ("val", "valid") else split
        
        # Load metadata
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Precompute valid track file pairs
        track_list = [
            (path / meta["audio_filename"][t], path / meta["midi_filename"][t])
            for t, s in meta["split"].items() # if s == split
        ]
        
        # Process tracks in parallel
        print(f"Loading Maestro ({split} split)...")
        with multiprocessing.Pool() as pool:
            results = list(tqdm(
                pool.imap(_process_track, track_list),
                total=len(track_list),
                desc="Processing tracks"
            ))
        
        # Filter invalid results
        data_list = [r for r in results if r is not None]
        
        super().__init__(data_list, **kwargs)