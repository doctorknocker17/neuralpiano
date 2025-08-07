import argparse

import soundfile as sf
import yaml
from importlib import import_module
import note_seq
from event_codec import Codec
from preprocessor import preprocess
from tqdm import tqdm

import matplotlib.pyplot as plt

import sys
import os

from diff import DiffusionLM

sys.path.append('/home/ubuntu/neuralpiano/neuralpiano/bigvgan/')
os.chdir('/home/ubuntu/neuralpiano/neuralpiano/bigvgan/')

import torch

from bigvgan import BigVGAN

os.chdir('/home/ubuntu/neuralpiano/neuralpiano/msd/')

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "nvidia/bigvgan_v2_44khz_128band_512x"
device = 'cuda'

@torch.no_grad()
def diff_main(model, tokens, segment_length, spec_frames, with_context, T=1000, verbose=True):
    """
    model: your transformer/diffusion spec-predictor
    tokens: iterable of token tensors
    segment_length: length of zero context in samples
    spec_frames: number of frames to predict per step
    with_context: bool, whether to carry mel_context
    """
    # --- Assume 'device' and 'vocoder' are defined in the outer scope ---
    output_specs = []
    zero_wav_context = torch.zeros(1, segment_length, device=device) if with_context else None
    mel_context = None

    # 2. Generate token-conditioned mel-specs
    for token in tqdm(tokens, disable=not verbose):
        x = token.unsqueeze(0).to(device)

        # pick which context arg to send
        if len(output_specs) == 0 and with_context:
            pred = model(
                x,
                seq_length=spec_frames,
                wav_context=zero_wav_context,
                rescale=False,
                T=T,
                verbose=verbose
            )
        else:
            pred = model(
                x,
                seq_length=spec_frames,
                mel_context=mel_context,
                rescale=False,
                T=T,
                verbose=verbose
            )

        output_specs.append(pred)
        mel_context = pred if with_context else None

    # stitch time-axis
    output_tensor = torch.cat(output_specs, dim=1)              # [1, total_frames, n_mels]

    # --- CRITICAL FIX HERE ---
    # Transpose to match MelToDB.reverse expected input shape [B, n_mels, T]
    output_tensor = output_tensor.transpose(-1, -2)              # [1, n_mels, total_frames]

    # --- CRITICAL CHANGE HERE ---
    # Use the model's MelToDB `reverse` method to convert the diffusion model's [-1, 1] output
    # back to the log-magnitude spectrogram format that the BigVGAN vocoder expects.
    # The updated `reverse` function handles the denormalization from [-1,1] to the log scale.
    log_mels_for_vocoder = model.mel[1].reverse(output_tensor)     # [1, n_mels, total_frames]
    # This `log_mels_for_vocoder` is now in the format: log(clamp(magnitude, min=clip_val))
    # which matches BigVGAN's `spectral_normalize_torch` output/input requirement.

    # Ensure correct dtype for BigVGAN (shape is already [B, n_mels, T])
    log_mels_for_vocoder = log_mels_for_vocoder.to(torch.float32) # Ensure float32

    # Optional: Add explicit clamp for numerical stability based on BigVGAN training stats
    # Typical log(magnitude) values are between -20 and 5. Adjust if needed based on observations.
    # log_mels_for_vocoder = torch.clamp(log_mels_for_vocoder, min=-30.0, max=10.0)

    # --- Generate waveform with BigVGAN ---
    with torch.inference_mode():
        wav_gen = vocoder(log_mels_for_vocoder)                 # [1, 1, T_audio]

    # --- Process output ---
    wav_gen_float = wav_gen.squeeze(0).squeeze(0).cpu()         # [T_audio]

    # BigVGAN typically outputs in [-1, 1] range, but let's be safe and clamp
    # Clamp to [-1, 1] range to prevent any potential clipping artifacts
    wav_gen_float = torch.clamp(wav_gen_float, -1.0, 1.0)

    # Convert to 16-bit PCM
    wav_int16 = (wav_gen_float * 32767.0).clamp(-32768, 32767).cpu().numpy().astype('int16')

    # Return the final audio and optionally the log-mel spectrogram used for vocoding
    # wav_int16 is np.ndarray with shape [T_audio] and int16 dtype
    # log_mels_for_vocoder.cpu().numpy()[0] is the log-mel spectrogram fed to the vocoder
    return wav_int16, log_mels_for_vocoder.cpu().numpy()[0]

    # If you want to return the intermediate linear magnitude spectrogram for analysis:
    # linear_mels = torch.exp(log_mels_for_vocoder)
    # return wav_int16, linear_mels.cpu().numpy()[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('midi', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('config', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-W', type=float, default=1.5
                       )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_configs = config['model']

    if args.W is not None:
        model_configs['init_args']['cfg_weighting'] = args.W

    # module_path, class_name = model_configs['class_path'].rsplit('.', 1)
    # module = import_module(module_path)
    module = import_module("diff")  # or the correct relative path
    model = getattr(module, "DiffusionLM").load_from_checkpoint(args.ckpt, **model_configs['init_args'])

        
    model = model.cuda()
    model.eval()

    hop_length = model_configs['init_args']['hop_length']
    n_mels = model_configs['init_args']['n_mels']
    data_configs = config['data']
    sr = data_configs['init_args']['sample_rate']
    segment_length = data_configs['init_args']['segment_length']
    spec_frames = segment_length // hop_length
    resolution = 100
    segment_length_in_time = segment_length / sr
    codec = Codec(int(segment_length_in_time * resolution + 1))

    with_context = data_configs['init_args']['with_context'] and model_configs['init_args']['with_context']

    print('With context:', with_context)

    ns = note_seq.midi_file_to_note_sequence(args.midi)
    ns = note_seq.apply_sustain_control_changes(ns)
    tokens, _ = preprocess(ns, codec=codec)
    
    vocoder = BigVGAN.from_pretrained(
            repo_id,
            use_cuda_kernel=False,
    )

    vocoder.h['fmax'] = 22050
    vocoder.h["num_freq"]   = 1025

    vocoder.cuda()
    vocoder.eval()
    vocoder.remove_weight_norm()

    pred, db_mels = diff_main(model, tokens, segment_length,
                     spec_frames, with_context)
    
    sf.write(args.output, pred, sr)

    plt.figure(figsize=(12, 4))
    plt.imshow(db_mels,
               aspect='auto',
               origin='lower',
               interpolation='nearest',
               cmap='magma')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Frame Index')
    plt.ylabel('Mel Bin Index')
    plt.title('Predicted Mel-Spectrogram')
    plt.tight_layout()
    
    plt.savefig('/home/ubuntu/mel_plot.png')
    plt.close()