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
  - µ-law companding to boost quiet passage resolution, value in the range 0-255
  - RMS normalization with soft limiting for optimal dynamic range
  - Three quality modes: 48kHz/21cy, 24kHz/42cy, 16kHz/63cy
  - Quality 1: 21-cycle exact unrolled playback loop (~82s)
  - Quality 2/3: JSR delay subroutine for exact 42/63-cycle timing (~165/248s)
  - Screen blanked during playback to prevent VIC-II badline cycle stealing

## General information
The script take most audiofiles and videofiles as input (whatever ffmeg supports).
Now featuring different samplerates to increase playtime at the expence of quality.
Have added 16KHz (1/3 samplerate) and 24Khz mode (1/2 samplerate)
  - quality 1 : 48KHz, aprox. 83s playback (default)
  - quality 2 : 24Khz, aprox. 161s playback
  - quality 3 : 16KHz, aprox. 248s playback

If the input audio is longer than what is supported within the limits of the 1MB cartridge and the given samplerate, the audio is cut.
µ-law usage seems to generate harsh audio with clipping. Needs more investigation

## Setup
1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
3. Install dependencies: `pip install -r requirements.txt`

External dependency: ffmpeg, install with apt/dnf, or other suitable packagemanagers

## Usage
```
usage: c64_easyflash_encoder.py [-h] [--sid {6581,8580}] [--ntsc] [--quality {1,2,3}] [--mu MU] input [output]

Arguments:
  input              Input audio file (MP3, WAV, FLAC, MP4, etc.)
  output             Output CRT file (default: output.crt)

options:
  -h, --help         show this help message and exit
  --sid {6581,8580}  SID chip model for amplitude table (default: 6581)
  --ntsc             Use NTSC clock (1022727 Hz) instead of PAL (985248 Hz)
  --quality {1,2,3}  1=48kHz/~82s 2=24kHz/~165s 3=16kHz/~248s (default: 1)
  --mu MU            µ-law companding coefficient, 0=off (default: 100)
```
