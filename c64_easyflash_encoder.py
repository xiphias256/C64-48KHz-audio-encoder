"""
MP3/WAV -> C64 EasyFlash CRT encoder (48 kHz VQ player)

Encodes audio into an EasyFlash cartridge image (.crt) that plays back
at ~47 kHz on a stock Commodore 64 using the Mahoney 8-bit D418 technique.

Key features:
  - 4:1 vector quantization with per-bank codebooks (256 x 4-sample vectors)
  - Two-stage VQ: DPCM-sqrt clustering for perceptual quality, centroids on A
  - Mahoney amplitude tables for both SID 6581 and 8580 (measured by Pex Tufvesson)
  - Quantization-aware refinement with TPDF dither
  - 21-cycle exact unrolled playback loop (all paths verified)
  - Pre-emphasis filter to boost HF content before VQ encoding

Usage:
  python c64_easyflash_encoder.py input.mp3 [output.crt] [--sid 6581|8580] [--ntsc]
"""

import sys
import math
import struct
import argparse
import numpy as np
from pydub import AudioSegment
from sklearn.cluster import KMeans
from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
K                     = 256          # codebook size (must be 256 for byte index)
VECTOR_SIZE           = 4            # samples per VQ vector
PAL_CLOCK             = 985248       # PAL C64 CPU clock (Hz)
NTSC_CLOCK            = 1022727      # NTSC C64 CPU clock (Hz)
CYCLES_PER_SAMPLE     = 21           # cycles between STA $D418 writes
CHIP_SIZE             = 0x2000       # 8 KB per CHIP packet
CODEBOOK_SIZE         = 0x0400       # 1 KB codebook (256 entries x 4 bytes)
ROML_STREAM_OFF       = 0x0400       # stream data starts after codebook in ROML
ROML_STREAM_LEN       = CHIP_SIZE - ROML_STREAM_OFF   # 7168 bytes
ROMH_STREAM_LEN       = CHIP_SIZE                     # 8192 bytes
STREAM_PER_BANK       = ROML_STREAM_LEN + ROMH_STREAM_LEN  # 15360 indices/bank
MAX_AUDIO_BANKS       = 63           # banks 1..63 (bank 0 = player)
PLAYER_RAM            = 0x0800       # player code destination in C64 RAM
STREAM_PAGES_ROML     = ROML_STREAM_LEN // 256   # 28
STREAM_PAGES_ROMH     = ROMH_STREAM_LEN // 256   # 32
STREAM_PAGES_PER_BANK = STREAM_PAGES_ROML + STREAM_PAGES_ROMH  # 60
STREAM_PAGE_HIS       = list(range(0x84, 0xA0)) + list(range(0xA0, 0xC0))
QUANT_REFINE_ROUNDS   = 5
PRE_EMPHASIS_COEFF    = 0            # high-pass pre-emphasis: y[n] = x[n] - coeff*x[n-1]
                                     # 0=disable , 0.95=full filter applied

# ---------------------------------------------------------------------------
# MAHONEY SID AMPLITUDE TABLES
# ---------------------------------------------------------------------------
# Measured output amplitudes for D418 values 0..255 with the Mahoney voice setup.
# Source: Musik_RunStop_8-bit_sample_measurements_by_Pex_Mahoney_Tufvesson.zip
#         Pex Mahoney Tufvesson, 25-Feb-2014

SID_AMPS_6581 = np.array([
    0.000086, 0.078433, 0.156209, 0.231024, 0.307690, 0.378565, 0.448696, 0.515601,
    0.603308, 0.665300, 0.726412, 0.784362, 0.843365, 0.897183, 0.950162, 1.000000,
   -0.004179,-0.024556,-0.044544,-0.063066,-0.081521,-0.098525,-0.115272,-0.130963,
   -0.151602,-0.166017,-0.180182,-0.193473,-0.206888,-0.219394,-0.231693,-0.243400,
   -0.000065, 0.078117, 0.155747, 0.230281, 0.306572, 0.377107, 0.446788, 0.513234,
    0.600298, 0.661856, 0.722471, 0.779941, 0.838519, 0.891821, 0.944257, 0.993670,
   -0.003885,-0.014557,-0.025069,-0.034769,-0.044310,-0.053237,-0.062063,-0.070251,
   -0.081120,-0.088694,-0.096163,-0.103111,-0.110014,-0.116566,-0.123005,-0.129079,
   -0.000598, 0.062600, 0.125077, 0.184902, 0.246071, 0.302364, 0.357843, 0.410717,
    0.480002, 0.528896, 0.576933, 0.622526, 0.669122, 0.711453, 0.753099, 0.792612,
   -0.004378,-0.025033,-0.045230,-0.063962,-0.082550,-0.099711,-0.116526,-0.132227,
   -0.153041,-0.167405,-0.181593,-0.194927,-0.208246,-0.220703,-0.233013,-0.244547,
   -0.000729, 0.064084, 0.128097, 0.189383, 0.251961, 0.309507, 0.366175, 0.420215,
    0.490810, 0.540626, 0.589607, 0.636005, 0.683332, 0.726454, 0.768713, 0.808805,
   -0.004135,-0.016167,-0.027995,-0.038889,-0.049613,-0.059619,-0.069502,-0.078647,
   -0.090824,-0.099261,-0.107537,-0.115289,-0.122993,-0.130225,-0.137417,-0.144156,
   -0.003176,-0.002153,-0.001294,-0.000331, 0.000786, 0.001658, 0.002413, 0.003247,
    0.004569, 0.005197, 0.005928, 0.006592, 0.007450, 0.008068, 0.008657, 0.009212,
   -0.008038,-0.084726,-0.158325,-0.226628,-0.294774,-0.356368,-0.416198,-0.472369,
   -0.547075,-0.598059,-0.647868,-0.694971,-0.742692,-0.786411,-0.829293,-0.869965,
   -0.002986, 0.007787, 0.018239, 0.028265, 0.038614, 0.047909, 0.057031, 0.065750,
    0.077309, 0.085296, 0.093201, 0.100736, 0.108584, 0.115538, 0.122572, 0.129133,
   -0.007022,-0.067578,-0.125870,-0.180039,-0.233970,-0.282941,-0.330587,-0.375266,
   -0.434452,-0.475052,-0.514810,-0.552379,-0.590348,-0.625363,-0.659726,-0.692313,
   -0.003658,-0.007161,-0.010732,-0.013928,-0.016903,-0.019846,-0.022738,-0.025347,
   -0.028787,-0.031180,-0.033594,-0.035799,-0.037789,-0.039903,-0.041917,-0.043782,
   -0.007618,-0.076203,-0.142110,-0.203299,-0.264221,-0.319423,-0.373066,-0.423379,
   -0.490150,-0.535812,-0.580477,-0.622698,-0.665400,-0.704636,-0.743203,-0.779710,
   -0.003445, 0.002169, 0.007504, 0.012721, 0.018201, 0.023043, 0.027707, 0.032291,
    0.038210, 0.042398, 0.046443, 0.050408, 0.054599, 0.058295, 0.061859, 0.065418,
   -0.006899,-0.062483,-0.115990,-0.165688,-0.215183,-0.260132,-0.303890,-0.344897,
   -0.399160,-0.436414,-0.472844,-0.507314,-0.542126,-0.574173,-0.605750,-0.635317,
], dtype=np.float32)

SID_AMPS_8580 = np.array([
    0.296841, 0.342174, 0.388113, 0.433498, 0.482807, 0.528693, 0.574186, 0.619396,
    0.676467, 0.722504, 0.767631, 0.812965, 0.863225, 0.908043, 0.953330, 1.000000,
    0.296585, 0.253814, 0.211532, 0.169380, 0.124805, 0.083160, 0.041534, 0.000451,
   -0.049665,-0.090874,-0.131023,-0.171055,-0.214557,-0.254048,-0.293259,-0.332582,
    0.296836, 0.341514, 0.386523, 0.431586, 0.480032, 0.524722, 0.570275, 0.614677,
    0.670630, 0.715622, 0.760787, 0.805202, 0.853724, 0.899193, 0.943526, 0.988241,
    0.296460, 0.253930, 0.211830, 0.169796, 0.125264, 0.083848, 0.042654, 0.001443,
   -0.048578,-0.089090,-0.129681,-0.169480,-0.212023,-0.252196,-0.291202,-0.330199,
    0.296649, 0.342981, 0.389236, 0.435599, 0.485816, 0.531765, 0.578136, 0.624629,
    0.682344, 0.728451, 0.775523, 0.821118, 0.871032, 0.917586, 0.963997, 1.009703,
    0.296347, 0.255303, 0.214417, 0.173850, 0.130589, 0.090463, 0.050517, 0.010497,
   -0.037865,-0.077090,-0.116374,-0.155345,-0.196354,-0.234968,-0.273508,-0.311138,
    0.296581, 0.342281, 0.387932, 0.433519, 0.482923, 0.528480, 0.574099, 0.619880,
    0.677029, 0.722269, 0.768037, 0.814036, 0.862826, 0.908465, 0.954833, 0.999726,
    0.296236, 0.255376, 0.214698, 0.174211, 0.131121, 0.091095, 0.051414, 0.011812,
   -0.036847,-0.075769,-0.114846,-0.153714,-0.194759,-0.232907,-0.271384,-0.309193,
    0.296631, 0.296051, 0.295420, 0.294675, 0.294027, 0.293355, 0.292672, 0.291986,
    0.291301, 0.290530, 0.289790, 0.289257, 0.288395, 0.287714, 0.287066, 0.286200,
    0.296024, 0.207189, 0.118702, 0.031332,-0.062120,-0.148408,-0.233674,-0.318311,
   -0.423175,-0.506204,-0.588661,-0.670987,-0.758686,-0.838896,-0.919460,-1.000000,
    0.296446, 0.295562, 0.294723, 0.293702, 0.292709, 0.291929, 0.290942, 0.289956,
    0.288819, 0.288125, 0.287000, 0.286121, 0.285209, 0.284152, 0.283296, 0.282474,
    0.296023, 0.207571, 0.119787, 0.032803,-0.059999,-0.145602,-0.230978,-0.314539,
   -0.418685,-0.501903,-0.583358,-0.664869,-0.752822,-0.832273,-0.911804,-0.991570,
    0.296631, 0.296919, 0.297435, 0.297892, 0.298243, 0.298690, 0.299148, 0.299392,
    0.299985, 0.300553, 0.300809, 0.301129, 0.301655, 0.302262, 0.302299, 0.302810,
    0.296092, 0.208841, 0.122328, 0.036532,-0.054881,-0.139210,-0.223246,-0.305961,
   -0.408225,-0.490167,-0.571972,-0.651102,-0.737570,-0.817478,-0.894767,-0.973014,
    0.296381, 0.296539, 0.296760, 0.297017, 0.297232, 0.297228, 0.297447, 0.297756,
    0.297718, 0.297945, 0.298221, 0.298231, 0.298337, 0.298736, 0.298706, 0.298886,
    0.295939, 0.209426, 0.123411, 0.038287,-0.053101,-0.136446,-0.219814,-0.302593,
   -0.403816,-0.485082,-0.566300,-0.646700,-0.730704,-0.810044,-0.889231,-0.964929,
], dtype=np.float32)

# Active LUTs — set by select_sid_model() before encoding
MAHONEY_LUT  = None   # linear index (0..255) -> D418 value
SID_AMP_NORM = None   # D418 value -> normalized float (-1..+1)

def build_lut(sid_amps):
    """Build forward and reverse LUTs from a Mahoney amplitude measurement table."""
    amp_min, amp_max = float(sid_amps.min()), float(sid_amps.max())
    targets = np.linspace(amp_min, amp_max, 256)
    lut = np.array([int(np.argmin(np.abs(sid_amps - t))) for t in targets],
                   dtype=np.uint8)
    amp_norm = (sid_amps - amp_min) / (amp_max - amp_min) * 2.0 - 1.0
    return lut, amp_norm

def select_sid_model(model):
    """Select SID model and build the corresponding amplitude LUTs."""
    global MAHONEY_LUT, SID_AMP_NORM
    amps = SID_AMPS_8580 if model == '8580' else SID_AMPS_6581
    MAHONEY_LUT, SID_AMP_NORM = build_lut(amps)
    print(f"  SID model: {model}")

# ---------------------------------------------------------------------------
# AUDIO LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def load_audio(path, sample_rate):
    """Load audio file, convert to mono, resample to target rate, normalize."""
    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)
    src_rate = audio.frame_rate
    print(f"  {audio.channels}ch {src_rate}Hz {audio.sample_width*8}-bit"
          f" -> {sample_rate}Hz mono")
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float64)
    # High-quality polyphase resampling via scipy
    if src_rate != sample_rate:
        g = math.gcd(sample_rate, src_rate)
        samples = resample_poly(samples, sample_rate // g, src_rate // g).astype(np.float64)
    # Normalize to -1..+1
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.98
    return samples.astype(np.float32)

def apply_pre_emphasis(samples):
    """Apply a simple first-order high-pass pre-emphasis filter.
    Boosts high frequencies before VQ encoding. The SID's natural
    low-pass rolloff partially compensates on playback, reducing
    perceived noise in the highs."""
    out = np.empty_like(samples)
    out[0] = samples[0]
    out[1:] = samples[1:] - PRE_EMPHASIS_COEFF * samples[:-1]
    # Re-normalize after pre-emphasis to avoid clipping
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.98
    return out

def make_vectors(samples):
    """Reshape sample array into (N, VECTOR_SIZE) matrix, trimming excess."""
    n = len(samples) // VECTOR_SIZE
    return samples[:n * VECTOR_SIZE].reshape(-1, VECTOR_SIZE)

# ---------------------------------------------------------------------------
# VECTOR QUANTIZATION
# ---------------------------------------------------------------------------

def companded_delta(signal):
    """Compute B = delta of sqrt-companded signal for perceptual clustering."""
    comp = np.sign(signal) * np.sqrt(np.abs(signal))
    return np.diff(comp, prepend=comp[0] if len(comp) else 0).astype(np.float32)

def recompute_centroids(vectors, labels):
    """Vectorized centroid computation using bincount (much faster than loop)."""
    cb = np.zeros((K, VECTOR_SIZE), dtype=np.float64)
    counts = np.bincount(labels, minlength=K)
    for d in range(VECTOR_SIZE):
        sums = np.bincount(labels, weights=vectors[:, d], minlength=K)
        mask = counts > 0
        cb[mask, d] = sums[mask] / counts[mask]
    return cb.astype(np.float32), int(np.sum(counts == 0))

def float_to_d418(cb):
    """Float centroids (-1..+1) -> D418 values via Mahoney LUT with TPDF dither."""
    # TPDF dither: triangular probability density, ±0.5 LSB
    dither = (np.random.random(cb.shape).astype(np.float32)
            + np.random.random(cb.shape).astype(np.float32) - 1.0) * (1.0 / 255.0)
    linear = np.clip(np.round((cb + dither + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return MAHONEY_LUT[linear]

def d418_to_float(d418_vals):
    """D418 values -> float (-1..+1) using measured SID amplitudes."""
    return SID_AMP_NORM[d418_vals]

def train_codebook(vectors, raw_signal, label=""):
    """Two-stage VQ: cluster on DPCM-sqrt(B), centroids on A, quant-aware refinement."""
    from scipy.spatial.distance import cdist

    # Step 1: companded DPCM
    b = companded_delta(raw_signal)
    bvecs = b[:len(vectors) * VECTOR_SIZE].reshape(-1, VECTOR_SIZE)

    # Step 2: cluster in B-space
    km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=0)
    km.fit(bvecs)

    # Step 3: recompute centroids on A
    cb, empty = recompute_centroids(vectors, km.labels_)

    # Step 4: quantization-aware refinement
    for _ in range(QUANT_REFINE_ROUNDS):
        cbq = float_to_d418(cb)
        cb_dq = d418_to_float(cbq)
        labels = np.argmin(cdist(vectors, cb_dq, metric='sqeuclidean'), axis=1).astype(np.int32)
        cb, empty = recompute_centroids(vectors, labels)

    # Final assignment with quantized codebook
    cbq = float_to_d418(cb)
    cb_dq = d418_to_float(cbq)
    labels = np.argmin(cdist(vectors, cb_dq, metric='sqeuclidean'), axis=1).astype(np.int32)

    # Report SNR
    reconst = cb_dq[labels].reshape(-1)
    sig_var = np.var(vectors)
    err_var = np.var(vectors.reshape(-1) - reconst)
    if err_var > 0:
        snr = 10 * np.log10(sig_var / err_var)
        if label:
            print(f"    {label}: SNR {snr:.1f} dB" + (f", {empty} empty" if empty else ""))

    return cb, labels

def interleave_codebook(cbq):
    """Interleave 4 codebook planes: plane[p] = cbq[:,p] for p in 0..3."""
    return np.concatenate([cbq[:, i] for i in range(VECTOR_SIZE)]).tobytes()

# ---------------------------------------------------------------------------
# 6502 ASSEMBLER HELPERS
# ---------------------------------------------------------------------------

class Asm:
    """Minimal 6502 assembler with label support."""
    def __init__(self, base):
        self.base = base
        self.buf = bytearray()
        self.labels = {}
        self.fixups = []
    def here(self):       return len(self.buf)
    def label(self, n):   self.labels[n] = self.here()
    def _b(self, *bs):    self.buf.extend(b & 0xFF for b in bs)
    def SEI(self):        self._b(0x78)
    def INX(self):        self._b(0xE8)
    def RTI(self):        self._b(0x40)
    def LDA_imm(self, v): self._b(0xA9, v)
    def LDA_abs(self, a): self._b(0xAD, a & 0xFF, a >> 8)
    def LDX_imm(self, v): self._b(0xA2, v)
    def LDX_zp(self, z):  self._b(0xA6, z)
    def STA_abs(self, a):  self._b(0x8D, a & 0xFF, a >> 8)
    def STA_zp(self, z):   self._b(0x85, z)
    def STX_abs(self, a):  self._b(0x8E, a & 0xFF, a >> 8)
    def STX_zp(self, z):   self._b(0x86, z)
    def CPX_imm(self, v):  self._b(0xE0, v)
    def JMP_abs(self, a):  self._b(0x4C, a & 0xFF, a >> 8)
    def JMP_label(self, n):
        self.fixups.append((self.here() + 1, n, 'j')); self._b(0x4C, 0, 0)
    def BNE(self, n):
        self.fixups.append((self.here() + 1, n, 'b')); self._b(0xD0, 0)
    def resolve(self):
        for off, name, kind in self.fixups:
            t = self.labels[name]
            if kind == 'j':
                a = self.base + t
                self.buf[off] = a & 0xFF; self.buf[off + 1] = a >> 8
            else:
                r = t - (off + 1)
                assert -128 <= r <= 127, f"Branch '{name}' out of range: {r}"
                self.buf[off] = r & 0xFF
    def bytes(self):
        self.resolve(); return bytes(self.buf)

# ---------------------------------------------------------------------------
# SID INIT (Mahoney white paper section XIV)
# ---------------------------------------------------------------------------

def emit_sid_init(a):
    """Configure SID voices for 8-bit sample playback via D418."""
    a.LDA_imm(0x0F)
    a.STA_abs(0xD405); a.STA_abs(0xD40C); a.STA_abs(0xD413)
    a.LDA_imm(0xFF)
    a.STA_abs(0xD406); a.STA_abs(0xD40D); a.STA_abs(0xD414)
    a.LDA_imm(0x49)
    a.STA_abs(0xD404); a.STA_abs(0xD40B); a.STA_abs(0xD412)
    a.LDA_imm(0xFF)
    a.STA_abs(0xD415); a.STA_abs(0xD416)
    a.LDA_imm(0x03); a.STA_abs(0xD417)
    a.LDA_imm(0x00); a.STA_abs(0xD418)

# ---------------------------------------------------------------------------
# UNROLLED PLAY BLOCKS (60 x 64 bytes = 3840 bytes)
# ---------------------------------------------------------------------------
# All paths between consecutive STA $D418 writes = exactly 21 cycles.

def make_unrolled_play_blocks(blocks_base, sid_addr, bank_done_addr):
    assert blocks_base % 64 == 0 and sid_addr % 256 == 0
    out = bytearray()
    for blk in range(STREAM_PAGES_PER_BANK):
        ph  = STREAM_PAGE_HIS[blk]
        ba  = blocks_base + blk * 64
        sm  = ba + 2
        nxt = (blocks_base + (blk + 1) * 64 + 1
               if blk < STREAM_PAGES_PER_BANK - 1
               else bank_done_addr)
        s = sid_addr
        b = bytearray(64)
        b[0]  = 0xEA
        b[1]  = 0xAE; b[2] = 0x00; b[3] = ph
        b[4]  = 0xBC; b[5] = 0x00; b[6] = 0x80
        b[7]  = 0xB9; b[8] = s & 0xFF; b[9] = s >> 8
        b[10] = 0x8D; b[11] = 0x18; b[12] = 0xD4
        b[13] = 0xEE; b[14] = sm & 0xFF; b[15] = sm >> 8
        b[16] = 0x24; b[17] = 0xEA
        b[18] = 0xBC; b[19] = 0x00; b[20] = 0x81
        b[21] = 0xB9; b[22] = s & 0xFF; b[23] = s >> 8
        b[24] = 0x8D; b[25] = 0x18; b[26] = 0xD4
        b[27] = 0xEE; b[28] = 0x20; b[29] = 0xD0
        b[30] = 0x24; b[31] = 0xEA
        b[32] = 0xBC; b[33] = 0x00; b[34] = 0x82
        b[35] = 0xB9; b[36] = s & 0xFF; b[37] = s >> 8
        b[38] = 0x8D; b[39] = 0x18; b[40] = 0xD4
        b[41] = 0x24; b[42] = 0xEA
        b[43] = 0xEA
        b[44] = 0xBC; b[45] = 0x00; b[46] = 0x83
        b[47] = 0xB9; b[48] = s & 0xFF; b[49] = s >> 8
        b[50] = 0xAC; b[51] = sm & 0xFF; b[52] = sm >> 8
        b[53] = 0x8D; b[54] = 0x18; b[55] = 0xD4
        b[56] = 0xD0; b[57] = (-58) & 0xFF
        b[58] = 0x4C; b[59] = nxt & 0xFF; b[60] = nxt >> 8
        b[61] = 0xEA; b[62] = 0xEA; b[63] = 0xEA
        out.extend(b)
    assert len(out) == STREAM_PAGES_PER_BANK * 64
    return bytes(out)

def make_bank_done_handler(blocks_base, addr, num_banks):
    h = Asm(addr)
    h.LDX_zp(0x02); h.INX(); h.CPX_imm(num_banks + 1)
    h.BNE('go')
    h.label('done'); h.JMP_label('done')
    h.label('go')
    h.STX_zp(0x02); h.STX_abs(0xDE00)
    for blk in range(STREAM_PAGES_PER_BANK):
        h.LDA_imm(0x00)
        h.STA_abs(blocks_base + blk * 64 + 2)
    h.JMP_abs(blocks_base + 1)
    return h.bytes()

def make_player_init(blocks_base, sid_addr):
    p = Asm(PLAYER_RAM)
    p.SEI()
    p.LDA_imm(0x07); p.STA_abs(0xDE02)
    p.LDA_imm(0x0B); p.STA_abs(0xD011)
    emit_sid_init(p)
    p.LDA_imm(1); p.STA_zp(0x02)
    p.LDA_imm(1); p.STA_abs(0xDE00)
    p.JMP_abs(blocks_base + 1)
    return p.bytes()

def build_player_blob(num_banks):
    """Assemble the complete player into a RAM blob ($0800 onward)."""
    BLOCKS = 0x0900
    BDONE  = BLOCKS + STREAM_PAGES_PER_BANK * 64
    SIDT   = ((BDONE + 512 + 255) & ~255)
    assert BLOCKS % 64 == 0 and SIDT % 256 == 0

    init   = make_player_init(BLOCKS, SIDT)
    blocks = make_unrolled_play_blocks(BLOCKS, SIDT, BDONE)
    done   = make_bank_done_handler(BLOCKS, BDONE, num_banks)
    sid    = bytes(range(256))  # identity sidtable

    assert len(done) <= 512

    BLOB_START = PLAYER_RAM
    BLOB_END   = SIDT + 256
    blob = bytearray(BLOB_END - BLOB_START)

    def place(addr, data):
        o = addr - BLOB_START
        assert 0 <= o and o + len(data) <= len(blob)
        blob[o:o + len(data)] = data

    place(PLAYER_RAM, init)
    place(BLOCKS, blocks)
    place(BDONE, done)
    place(SIDT, sid)
    return bytes(blob), BLOCKS, BDONE, SIDT

# ---------------------------------------------------------------------------
# COPY STUB & KERNAL REPLACEMENT (EasyFlash boot)
# ---------------------------------------------------------------------------

def make_copy_stub(player_size):
    pages = math.ceil(player_size / 256)
    assert pages <= 20
    stub = bytearray([0xA2, 0x00])
    loop = len(stub)
    for p in range(pages):
        stub.extend([0xBD, 0x80, 0x80 + p, 0x9D, 0x00, 0x08 + p])
    stub.append(0xE8)
    stub.extend([0xD0, (loop - len(stub) - 2) & 0xFF])
    stub.extend([0x4C, 0x00, 0x08])
    stub.extend(b'\x00' * (0x80 - len(stub)))
    return bytes(stub[:0x80])

def make_kernal_romh():
    romh = bytearray(CHIP_SIZE)
    code = bytearray([
        0x78, 0xA2, 0xFF, 0x9A, 0xD8,
        0xA9, 0x08, 0x8D, 0x16, 0xD0,
        0x9D, 0x00, 0x01, 0xCA, 0xD0, 0xFB,
        0xA2, 0x3E,
    ])
    loop_off = len(code)
    code.extend([0xBD, 0x1E, 0xE0, 0x9D, 0x00, 0x01, 0xCA])
    code.extend([0x10, (loop_off - len(code) - 2) & 0xFF, 0x4C, 0x00, 0x01])
    while len(code) < 0x1E:
        code.insert(-3, 0xEA)
    romh[0:len(code)] = code
    stub = bytearray([
        0xA9, 0x87, 0x8D, 0x02, 0xDE,
        0xA9, 0x7F, 0x8D, 0x00, 0xDC,
        0xA2, 0xFF, 0x8E, 0x02, 0xDC,
        0xE8, 0x8E, 0x03, 0xDC,
        0xAD, 0x01, 0xDC,
        0x8E, 0x02, 0xDC, 0x8E, 0x00, 0xDC,
        0x29, 0xE0, 0xC9, 0xE0,
    ])
    bne_pos = len(stub)
    stub.extend([0xD0, 0x00])
    stub.extend([
        0xA2, 0x00, 0x8E, 0x16, 0xD0,
        0x20, 0x84, 0xFF, 0x4E, 0x87, 0xFF,
        0x20, 0x8A, 0xFF, 0x20, 0x81, 0xFF,
        0x4C, 0x00, 0x80,
    ])
    stub[bne_pos + 1] = (len(stub) - bne_pos - 2) & 0xFF
    stub.extend([0xA9, 0x04, 0x8D, 0x02, 0xDE, 0x6C, 0xFC, 0xFF, 0x00])
    stub.extend(b'\x00' * (0x3F - len(stub)))
    romh[0x1E:0x1E + 0x3F] = stub[:0x3F]
    romh[0x1F40] = 0x40
    romh[0x1FFA] = 0xFE; romh[0x1FFB] = 0xFF
    romh[0x1FFC] = 0x00; romh[0x1FFD] = 0xE0
    romh[0x1FFE] = 0x40; romh[0x1FFF] = 0xFF
    return bytes(romh)

# ---------------------------------------------------------------------------
# CRT FORMAT
# ---------------------------------------------------------------------------

def crt_header():
    h = bytearray(0x40)
    h[0:16] = b'C64 CARTRIDGE   '
    struct.pack_into('>I', h, 0x10, 0x40)
    struct.pack_into('>H', h, 0x14, 0x0100)
    struct.pack_into('>H', h, 0x16, 0x0020)  # EasyFlash
    h[0x18] = 0x01; h[0x19] = 0x00
    h[0x20:0x40] = b'C64 48KHZ VQ PLAYER'.ljust(0x20, b'\x00')
    return bytes(h)

def chip_packet(bank, load_addr, data):
    assert len(data) == CHIP_SIZE
    hdr = bytearray(0x10)
    hdr[0:4] = b'CHIP'
    struct.pack_into('>I', hdr, 4, CHIP_SIZE + 0x10)
    struct.pack_into('>H', hdr, 8, 0x0002)
    struct.pack_into('>H', hdr, 10, bank)
    struct.pack_into('>H', hdr, 12, load_addr)
    struct.pack_into('>H', hdr, 14, CHIP_SIZE)
    return bytes(hdr) + data

def assemble_audio_bank(cb_bytes, stream):
    roml = bytearray(CHIP_SIZE)
    romh = bytearray(CHIP_SIZE)
    roml[0:CODEBOOK_SIZE] = cb_bytes
    roml[ROML_STREAM_OFF:ROML_STREAM_OFF + ROML_STREAM_LEN] = bytes(stream[:ROML_STREAM_LEN])
    romh[0:min(len(stream) - ROML_STREAM_LEN, ROMH_STREAM_LEN)] = bytes(
        stream[ROML_STREAM_LEN:ROML_STREAM_LEN + ROMH_STREAM_LEN])
    return bytes(roml), bytes(romh)

# ---------------------------------------------------------------------------
# ENCODER
# ---------------------------------------------------------------------------

def encode_to_crt(input_file, output_file="output.crt", sid_model="6581", ntsc=False):
    clock = NTSC_CLOCK if ntsc else PAL_CLOCK
    sample_rate = clock // CYCLES_PER_SAMPLE
    system = "NTSC" if ntsc else "PAL"
    print(f"System: {system} ({clock} Hz clock, {sample_rate} Hz sample rate)")

    select_sid_model(sid_model)
    print(f"Loading: {input_file}")
    samples = load_audio(input_file, sample_rate)
    duration = len(samples) / sample_rate
    print(f"  {len(samples)} samples ({duration:.1f}s)")

    # Pre-emphasis
    samples = apply_pre_emphasis(samples)

    # Segment into banks
    all_vecs = make_vectors(samples)
    max_vecs = MAX_AUDIO_BANKS * STREAM_PER_BANK
    if len(all_vecs) > max_vecs:
        max_dur = max_vecs * VECTOR_SIZE / sample_rate
        print(f"  Warning: truncating to {max_dur:.1f}s")
        all_vecs = all_vecs[:max_vecs]
    total_vecs = len(all_vecs)
    n_banks = math.ceil(total_vecs / STREAM_PER_BANK)
    audio_dur = total_vecs * VECTOR_SIZE / sample_rate
    print(f"  {total_vecs} vectors -> {n_banks} banks ({audio_dur:.1f}s)")

    # Build player
    print("Building player...")
    blob, blocks, bdone, sidt = build_player_blob(n_banks)
    print(f"  {len(blob)} bytes, blocks=${blocks:04X} done=${bdone:04X} sid=${sidt:04X}")

    stub = make_copy_stub(len(blob))
    roml0 = bytearray(CHIP_SIZE)
    roml0[0:0x80] = stub
    pages = math.ceil(len(blob) / 256)
    for pg in range(pages):
        chunk = blob[pg * 256:(pg + 1) * 256]
        roml0[pg * 256 + 0x80:pg * 256 + 0x80 + len(chunk)] = chunk

    # Assemble CRT
    print(f"Training {n_banks} per-bank codebooks...")
    crt = bytearray(crt_header())
    crt += chip_packet(0, 0x8000, bytes(roml0))
    crt += chip_packet(0, 0xA000, make_kernal_romh())

    raw_aligned = samples[:total_vecs * VECTOR_SIZE]
    for i in range(n_banks):
        vs = i * STREAM_PER_BANK
        ve = min(vs + STREAM_PER_BANK, total_vecs)
        bank_vecs = all_vecs[vs:ve]
        bank_raw  = raw_aligned[vs * VECTOR_SIZE:ve * VECTOR_SIZE]

        cb, labels = train_codebook(bank_vecs, bank_raw,
                                    label=f"bank {i+1}/{n_banks}")
        cbq = float_to_d418(cb)
        cbb = interleave_codebook(cbq)
        stream = labels.astype(np.uint8)
        if len(stream) < STREAM_PER_BANK:
            stream = np.pad(stream, (0, STREAM_PER_BANK - len(stream)))

        ra, rh = assemble_audio_bank(cbb, stream)
        crt += chip_packet(i + 1, 0x8000, ra)
        crt += chip_packet(i + 1, 0xA000, rh)

    with open(output_file, "wb") as f:
        f.write(crt)
    print(f"Saved: {output_file} ({len(crt)//1024} KB, {n_banks+1} banks, {audio_dur:.1f}s)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Encode audio into a C64 EasyFlash CRT (48 kHz VQ player)")
    p.add_argument("input", help="Input audio file (MP3, WAV, FLAC, etc.)")
    p.add_argument("output", nargs="?", default="output.crt",
                   help="Output CRT file (default: output.crt)")
    p.add_argument("--sid", choices=["6581", "8580"], default="6581",
                   help="SID chip model for amplitude table (default: 6581)")
    p.add_argument("--ntsc", action="store_true",
                   help="Use NTSC clock (1022727 Hz) instead of PAL (985248 Hz)")
    args = p.parse_args()
    encode_to_crt(args.input, args.output, sid_model=args.sid, ntsc=args.ntsc)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python c64_easyflash_encoder.py input [output.crt]"
              " [--sid 6581|8580] [--ntsc]")
        sys.exit(0)
    main()
