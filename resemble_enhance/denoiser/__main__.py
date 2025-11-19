import argparse
from pathlib import Path

import torch
import torchaudio

from .inference import denoise


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=Path, help="Path to input audio file")
    parser.add_argument("out_file", type=Path, help="Output file")
    parser.add_argument("--run_dir", type=Path, default="runs/denoiser", help="Path to run folder")
    parser.add_argument("--suffix", type=str, default=".wav", help="File suffix")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    #for path in args.in_dir.glob(f"**/*{args.suffix}"):
    path = args.in_file
    print(f"Processing {path} ..")
    dwav, sr = torchaudio.load(path)
    hwav, sr = denoise(dwav[0], sr, args.run_dir, args.device)
    out_path = args.out_file
    #out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path, hwav[None], sr)


if __name__ == "__main__":
    main()
