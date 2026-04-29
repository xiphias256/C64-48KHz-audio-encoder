# C64 48 KHz Digital Audio Encoder

Encodes audio into an EasyFlash cartridge image (.crt) that plays back at up to ~48 kHz on a stock Commodore 64 using the SID chip's D418 volume register technique.

This Python script replicates and extends the work of **Antonio Savona (tonysavon)**, who created the original 48 kHz HiFi digital music players for the C64:

- [C64 48 KHz HiFi Digi Player 1](https://csdb.dk/release/?id=162941)
- [C64 48 KHz HiFi Digi Player 2](https://csdb.dk/release/?id=162951)
- [Technical writeup](https://brokenbytes.blogspot.com/2018/03/a-48khz-digital-music-player-for.html)

The playback technique is based on **Pex "Mahoney" Tufvesson's** groundbreaking work on 8-bit sample playback through the SID's volume register, documented in his [RUN/STOP technical details (PDF)](https://livet.se/mahoney/c64-files/Musik_RunStop_Technical_Details_by_Pex_Mahoney_Tufvesson_v2.pdf).

## How It Works

The encoder uses **vector quantization (VQ)** to compress audio into a format that the C64 can decompress and play in real time. Each audio sample is written to the SID chip's volume register ($D418) at a fixed rate, producing analog audio output through the chip's DAC coupling.

The playback loop is cycle-exact — every path between consecutive D418 writes takes the same number of CPU cycles, ensuring jitter-free output. The screen is blanked during playback to prevent VIC-II badline cycle stealing.

## Features

### Encoding
- **4:1 vector quantization** with per-bank codebooks (256 entries × 4 samples each)
- **Two-stage VQ pipeline**: clusters on DPCM-sqrt signal for perceptual weighting, recomputes centroids on the original waveform (as described in tonysavon's writeup)
- **Quantization-aware refinement** (5 rounds) with TPDF dither to minimize encode/decode mismatch
- **Mahoney amplitude tables** for both SID 6581 and 8580, measured by Pex Tufvesson (2014)
- **µ-law companding** (optional) to boost quiet passage resolution
- **High-quality resampling** via scipy polyphase filter
- Accepts any audio/video format supported by ffmpeg

### Playback
- **Three quality/duration modes** on a 1 MB EasyFlash cartridge:

  | Quality | Cycles | Sample Rate | Max Duration | Player Design |
  |---------|--------|-------------|--------------|---------------|
  | 1 (default) | 21 | ~47 kHz | ~83 s | 60 unrolled play blocks |
  | 2 | 41 | ~24 kHz | ~161 s | Unrolled blocks + JSR delay |
  | 3 | 63 | ~16 kHz | ~248 s | Unrolled blocks + JSR delay |

- **Cycle-exact timing** on all 5 inter-sample paths (verified)
- **Screen blanking** to eliminate VIC-II badline interference
- Supports both **PAL** (985248 Hz) and **NTSC** (1022727 Hz) clock rates

## Setup

### Prerequisites
- Python 3.8+
- ffmpeg (install via your system package manager)

### Installation
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# .\venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Usage

```
python c64_easyflash_encoder.py input [output] [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input audio or video file (MP3, WAV, FLAC, MP4, etc.) |
| `output` | Output CRT file (default: `output.crt`) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sid {6581,8580}` | `6581` | SID chip model for the Mahoney amplitude table |
| `--ntsc` | off | Use NTSC clock instead of PAL |
| `--quality {1,2,3}` | `1` | Sample rate / duration tradeoff (see table above) |
| `--mu MU` | `0` | µ-law companding coefficient (0 = off, 255 = maximum) |

### Examples

```bash
# Default: 48 kHz, SID 6581, PAL
python c64_easyflash_encoder.py song.mp3 song.crt

# Lower quality for longer playback
python c64_easyflash_encoder.py song.mp3 song.crt --quality 3

# For SID 8580 machines
python c64_easyflash_encoder.py song.mp3 song.crt --sid 8580

# NTSC with mild companding for quieter passages
python c64_easyflash_encoder.py song.mp3 song.crt --ntsc --mu 10

# Extract audio from a video file
python c64_easyflash_encoder.py concert.mp4 concert.crt --quality 2
```

## Notes

- If the input audio exceeds the maximum duration for the selected quality mode, it is truncated to fit the 1 MB cartridge.
- The `--mu` parameter controls µ-law companding intensity. Higher values boost quiet passages more aggressively, which can improve perceived quality on material with wide dynamic range. Off by default. Values above 10 tend to cause distortion and clipping — try `--mu 5` or `--mu 10` as a starting point.
- The `--sid` option selects which Mahoney amplitude lookup table to bake into the cartridge. Use `6581` for the original SID chip and `8580` for the later revision. Using the wrong table will produce distorted audio on real hardware. In VICE, match the SID model configured in Settings → SID.
- Quality 2 and 3 use a JSR delay subroutine to extend the cycle count between D418 writes. The audio is resampled to the lower rate, so pitch remains correct.

## Credits

- **Antonio Savona (tonysavon)** — original 48 kHz VQ player design and encoder algorithm
- **Pex "Mahoney" Tufvesson** — SID D418 8-bit sample playback technique and amplitude measurements
- **Encoder script** — developed with assistance from Claude (Anthropic)
