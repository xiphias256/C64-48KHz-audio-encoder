# C64-48KHz-audio-encoder
Encodes audio into an EasyFlash cartridge image (.crt) (1MB) that plays back at ~48 kHz on a stock Commodore 64 or Vice emulator

This Python script is made to replicate the excellent work of Antonio Savona, who made
- [C64 48Khz HiFi Digi Player 1](https://csdb.dk/release/?id=162941)
- [C64 48Khz HiFi Digi Player 2](https://csdb.dk/release/?id=162951)

His writeup of how he managed to make those cartridges is found [here](https://brokenbytes.blogspot.com/2018/03/a-48khz-digital-music-player-for.html).

The technical description of the C64 demo RUN/STOP by Pex 'Mahoney' Tufvesson, 
was also crutial in understanding how to manipulate the SID chip. 
His writeup of the RUN/STOP demo is found [here](https://livet.se/mahoney/c64-files/Musik_RunStop_Technical_Details_by_Pex_Mahoney_Tufvesson_v2.pdf).

## Key features of the script:
  - 4:1 vector quantization with per-bank codebooks (256 x 4-sample vectors)
  - Two-stage VQ: DPCM-sqrt clustering for perceptual quality, centroids on A
  - Mahoney amplitude tables for both SID 6581 and 8580 (measured by Pex Tufvesson)
  - Quantization-aware refinement with TPDF dither
  - 21-cycle exact unrolled playback loop (all paths verified)
  - Pre-emphasis filter to boost HF content before VQ encoding

## Setup
1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
3. Install dependencies: `pip install -r requirements.txt`

## Usage
```
c64_easyflash_encoder.py [-h] [--sid {6581,8580}] [--ntsc] input [output]

Arguments:
  input              Input audio file (MP3, WAV, FLAC, etc.)
  output             Output CRT file (default: output.crt)

Options:
  -h, --help         show this help message and exit
  --sid {6581,8580}  SID chip model for amplitude table (default: 6581)
  --ntsc             Use NTSC clock (1022727 Hz) instead of PAL (985248 Hz)
```
