#!/usr/bin/env python3
"""De-limiter Studio CLI & Audio Engine (macOS, Linux, Windows).

Can be run directly from the terminal on any platform with the same processing,
loudness rebalancing, and format conversion options & defaults as the macOS GUI:

  python delimiter_tool.py [options] INPUT [INPUT ...]

Examples:
  # Default GUI settings: CPU, Album Relative Rebalance (-14 LUFS / -1 dBTP), 24-bit FLAC
  python delimiter_tool.py ./MyAlbum

  # NVIDIA GPU (Linux/Windows) or Apple Silicon GPU (macOS), 128 kbps Opus output
  python delimiter_tool.py ./MyLibrary --device auto --format opus --bitrate 128

  # Custom target loudness, ceiling, and output directory
  python delimiter_tool.py track1.wav track2.wav -f flac -l -12.0 -c -0.5 -o ./Mastered

Internal GUI helper subcommands (used by De-limiter.app, emit `RESULT <json>`):
  stage SRC DST
  measure FILE [FILE ...]
  write SRC DST --format wav|flac|m4a|opus [--bitrate KBPS] [--gain DB] [--tags-from FILE]
"""
import argparse
import base64
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import pyloudnorm
import scipy.signal
import soundfile as sf
import soxr

MODEL_RATE = 44100
OPUS_RATE = 48000
WAVEFORM_BUCKETS = 480
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif", ".ogg", ".opus"}
FORMAT_EXTENSIONS = {
    "wav": "wav",
    "flac": "flac",
    "aac": "m4a",
    "m4a": "m4a",
    "opus": "opus",
}
DEFAULT_BITRATES = {
    "aac": 256,
    "m4a": 256,
    "opus": 192,
}
TAG_FIELDS = ("title", "artist", "album", "tracknumber", "date", "genre")

# Keys in Core Audio's info dictionary (`afinfo -i`) -> libsndfile tag names
CORE_AUDIO_TAGS = {
    "title": "title",
    "artist": "artist",
    "album": "album",
    "track number": "tracknumber",
    "year": "date",
    "recorded date": "date",
    "genre": "genre",
}

# Keys in ffprobe's format.tags -> libsndfile tag names
FFPROBE_TAGS = {
    "title": "title",
    "artist": "artist",
    "album": "album",
    "track": "tracknumber",
    "tracknumber": "tracknumber",
    "date": "date",
    "year": "date",
    "genre": "genre",
}

_OGG_CRC_TABLE = [0] * 256
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if (_r & 0x80000000) else ((_r << 1) & 0xFFFFFFFF)
    _OGG_CRC_TABLE[_i] = _r


def _ogg_crc32(data):
    """Standard Ogg framing CRC-32 (polynomial 0x04C11DB7, direct, initial 0, no final XOR)."""
    crc = 0
    tbl = _OGG_CRC_TABLE
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ tbl[((crc >> 24) ^ byte) & 0xFF]
    return crc


# ---------------------------------------------------------------------------
# Cross-Platform Audio Decoding & Loudness Metering
# ---------------------------------------------------------------------------

def read_audio(path):
    """Returns (float32 array shaped frames x channels, sample rate)."""
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        return data, rate
    except RuntimeError:
        pass

    # Fallback for containers libsndfile doesn't open (e.g. M4A / AAC / ALAC):
    # use macOS built-in afconvert when available, or ffmpeg on Linux / Windows.
    with tempfile.TemporaryDirectory() as tmp:
        decoded = os.path.join(tmp, "decoded.wav")
        if sys.platform == "darwin" and os.path.isfile("/usr/bin/afconvert"):
            res = subprocess.run(
                ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEF32", path, decoded],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                return sf.read(decoded, dtype="float32", always_2d=True)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            res = subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", path, "-f", "wav", "-acodec", "pcm_f32le", decoded],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                return sf.read(decoded, dtype="float32", always_2d=True)
            raise RuntimeError(f"unsupported or unreadable audio file ({res.stderr.strip() or 'ffmpeg decode failed'})")

        raise RuntimeError(
            "unsupported or unreadable audio file (install ffmpeg on PATH to decode M4A/AAC on Linux/Windows)"
        )


def to_stereo(data):
    """The model is stereo-only: duplicate mono, fold multichannel down to L/R."""
    channels = data.shape[1]
    if channels == 2:
        return data
    if channels == 1:
        return np.repeat(data, 2, axis=1)
    left, right = data[:, 0].copy(), data[:, 1].copy()
    if channels >= 3:  # front centre at -3 dB into both sides
        left += 0.7071 * data[:, 2]
        right += 0.7071 * data[:, 2]
    if channels >= 6:  # 5.1 order FL FR FC LFE BL BR: surrounds at -3 dB
        left += 0.7071 * data[:, 4]
        right += 0.7071 * data[:, 5]
    return np.stack([left, right], axis=1)


def true_peak(data, rate, block_seconds=10):
    """Peak of the 4x-oversampled signal, streamed in blocks so long tracks don't need 4x the memory."""
    stream = soxr.ResampleStream(rate, rate * 4, data.shape[1], dtype="float32", quality="HQ")
    peak, block = 0.0, rate * block_seconds
    for start in range(0, len(data), block):
        chunk = np.ascontiguousarray(data[start:start + block])
        oversampled = stream.resample_chunk(chunk, last=start + block >= len(data))
        if oversampled.size:
            peak = max(peak, float(np.abs(oversampled).max()))
    return peak


def core_loudness(data, rate):
    """Sustained 'core' loudness (LUFS): 10th-60th percentile of active 10 ms K-weighted micro-frames
    inside each 400 ms window with ITU-R BS.1770-4 two-stage gating, ignoring transient spikes."""
    f0_1, gain_1, q_1 = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k1 = math.tan(math.pi * f0_1 / rate)
    vh = 10.0 ** (gain_1 / 20.0)
    vb = vh ** 0.4996667741545416
    k1_2 = k1 * k1
    a0_1 = 1.0 + k1 / q_1 + k1_2
    b1 = [(vh + vb * k1 / q_1 + k1_2) / a0_1, 2.0 * (k1_2 - vh) / a0_1, (vh - vb * k1 / q_1 + k1_2) / a0_1]
    a1 = [1.0, 2.0 * (k1_2 - 1.0) / a0_1, (1.0 - k1 / q_1 + k1_2) / a0_1]

    f0_2, q_2 = 38.13547087602444, 0.5003270373238773
    k2 = math.tan(math.pi * f0_2 / rate)
    k2_2 = k2 * k2
    a0_2 = 1.0 + k2 / q_2 + k2_2
    b2 = [1.0 / a0_2, -2.0 / a0_2, 1.0 / a0_2]
    a2 = [1.0, 2.0 * (k2_2 - 1.0) / a0_2, (1.0 - k2 / q_2 + k2_2) / a0_2]

    sos = np.array([b1 + a1, b2 + a2], dtype=np.float32)
    micro_step = max(1, round(0.010 * rate))
    n_micro = len(data) // micro_step
    if n_micro < 40:
        return -math.inf

    k_weighted = scipy.signal.sosfilt(sos, data[:n_micro * micro_step], axis=0)
    micro_energy = np.mean(
        k_weighted.reshape(n_micro, micro_step, data.shape[1]).astype(np.float64) ** 2,
        axis=1,
    ).sum(axis=1)
    windows = sliding_window_view(micro_energy, 40)[::10]
    win_mean = windows.mean(axis=1)
    sorted_w = np.sort(windows, axis=1)

    core_energies = np.empty(len(windows), dtype=np.float64)
    for i in range(len(windows)):
        m = win_mean[i]
        if m <= 1e-12:
            core_energies[i] = m
            continue
        row = sorted_w[i]
        idx0 = int(np.searchsorted(row, m * 0.0316))
        active_cnt = 40 - idx0
        if active_cnt < 4:
            core_energies[i] = m
        else:
            lo = idx0 + int(active_cnt * 0.10)
            hi = max(lo + 1, idx0 + math.ceil(active_cnt * 0.60))
            core_energies[i] = row[lo:hi].mean()

    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    abs_gated = core_energies[core_energies > abs_gate]
    if len(abs_gated) == 0:
        return -math.inf
    rel_lufs = -0.691 + 10.0 * math.log10(abs_gated.mean()) - 10.0
    rel_gate = 10.0 ** ((rel_lufs + 0.691) / 10.0)
    rel_gated = abs_gated[abs_gated > rel_gate]
    if len(rel_gated) == 0:
        return -0.691 + 10.0 * math.log10(abs_gated.mean())
    return -0.691 + 10.0 * math.log10(rel_gated.mean())


def waveform_peaks(data, num_buckets=WAVEFORM_BUCKETS):
    """Downsamples stereo float audio into `num_buckets` peak amplitude values in [0, 1]."""
    n = len(data)
    if n == 0 or num_buckets <= 0:
        return []
    actual = min(num_buckets, n)
    edges = np.linspace(0, n, actual + 1, dtype=np.int64)
    abs_max_ch = np.max(np.abs(data), axis=1)
    peaks = np.maximum.reduceat(abs_max_ch, edges[:-1])
    return [round(float(p), 4) for p in peaks]


def loudness(data, rate, include_peaks=True):
    try:
        lufs = pyloudnorm.Meter(rate).integrated_loudness(data)
    except ValueError:  # shorter than one 400 ms gating block
        lufs = -math.inf
    core = core_loudness(data, rate)
    peak = true_peak(data, rate)
    peak_db = 20 * math.log10(peak) if peak > 0 else -math.inf
    result = {
        "lufs": round(lufs, 3) if math.isfinite(lufs) else None,
        "core_lufs": round(core, 3) if math.isfinite(core) else None,
        "peak": round(peak_db, 3) if math.isfinite(peak_db) else None,
    }
    if include_peaks:
        result["peaks"] = waveform_peaks(data)
    return result


# ---------------------------------------------------------------------------
# Metadata Reading & Tagging (FLAC, Opus, M4A, WAV)
# ---------------------------------------------------------------------------

def read_tags(path):
    """Basic tags (title, artist, album, tracknumber, date, genre) from the source file."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with sf.SoundFile(path) as source:
            metadata = source.copy_metadata()
        return {key: value for key, value in metadata.items() if key in TAG_FIELDS and value}
    except RuntimeError:
        pass

    # Fallback for M4A: Core Audio afinfo on macOS, ffprobe on Linux/Windows
    if sys.platform == "darwin" and os.path.isfile("/usr/bin/afinfo"):
        info = subprocess.run(["/usr/bin/afinfo", "-i", path], capture_output=True, text=True).stdout
        tags, in_dictionary = {}, False
        for line in info.splitlines():
            if line.strip() == "info dictionary:":
                in_dictionary = True
                continue
            if in_dictionary:
                match = re.match(r'^\s+(.+?)\s*:\s*"(.*)"\s*$', line)
                if not match:
                    break
                key = CORE_AUDIO_TAGS.get(match.group(1).strip())
                if key and match.group(2) and key not in tags:
                    tags[key] = match.group(2)
        if tags:
            return tags

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        res = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            try:
                raw_tags = json.loads(res.stdout).get("format", {}).get("tags", {})
            except ValueError:
                raw_tags = {}
            tags = {}
            for k, v in raw_tags.items():
                mapped = FFPROBE_TAGS.get(k.lower().strip())
                if mapped and v and mapped not in tags:
                    tags[mapped] = str(v)
            return tags

    return {}


def extract_flac_tags_and_pictures(flac_path):
    """Extracts Vorbis comments (excluding stale ReplayGain/R128/albumversion) and PICTURE blocks from a FLAC file."""
    comments, pictures = [], []
    if not flac_path or not os.path.isfile(flac_path):
        return comments, pictures
    with open(flac_path, "rb") as f:
        if f.read(4) != b"fLaC":
            return comments, pictures
        while True:
            bhdr = f.read(4)
            if len(bhdr) < 4:
                break
            is_last = (bhdr[0] & 0x80) != 0
            btype = bhdr[0] & 0x7F
            blen = int.from_bytes(bhdr[1:4], "big")
            bdata = f.read(blen)
            if len(bdata) < blen:
                break
            if btype == 4 and len(bdata) >= 8:
                vlen = struct.unpack_from("<I", bdata, 0)[0]
                cpos = 4 + vlen
                if cpos + 4 <= len(bdata):
                    cnt = struct.unpack_from("<I", bdata, cpos)[0]
                    cur = cpos + 4
                    for _ in range(cnt):
                        if cur + 4 > len(bdata):
                            break
                        elen = struct.unpack_from("<I", bdata, cur)[0]
                        entry = bdata[cur + 4:cur + 4 + elen].decode("utf-8", errors="replace")
                        cur += 4 + elen
                        key_upper = entry.split("=", 1)[0].strip().upper()
                        if not key_upper.startswith(("REPLAYGAIN_", "R128_")) and key_upper != "ALBUMVERSION":
                            comments.append(entry)
            elif btype == 6:
                pictures.append(bdata)
            if is_last:
                break
    return comments, pictures


def rewrite_flac_metadata(path, comments, pictures=()):
    """Rewrites a FLAC file's VORBIS_COMMENT and PICTURE blocks while keeping audio frames intact."""
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != b"fLaC":
        return
    pos = 4
    kept_blocks = []
    vendor = b"De-limiter Studio"
    while pos + 4 <= len(buf):
        hdr = buf[pos]
        is_last = (hdr & 0x80) != 0
        btype = hdr & 0x7F
        blen = int.from_bytes(buf[pos + 1:pos + 4], "big")
        bdata = buf[pos + 4:pos + 4 + blen]
        pos += 4 + blen
        if btype == 4 and len(bdata) >= 4:
            vlen = struct.unpack_from("<I", bdata, 0)[0]
            if 4 + vlen <= len(bdata):
                vendor = bdata[4:4 + vlen]
        elif btype not in (1, 4, 6):
            kept_blocks.append((btype, bdata))
        if is_last:
            break
    audio_frames = buf[pos:]

    vc = bytearray(struct.pack("<I", len(vendor)) + vendor)
    encoded = [c.encode("utf-8") for c in comments]
    vc += struct.pack("<I", len(encoded))
    for entry in encoded:
        vc += struct.pack("<I", len(entry)) + entry
    if len(vc) < (1 << 24):
        kept_blocks.append((4, bytes(vc)))
    for pic in pictures:
        if len(pic) < (1 << 24):
            kept_blocks.append((6, pic))

    out = bytearray(b"fLaC")
    for idx, (btype, bdata) in enumerate(kept_blocks):
        last_flag = 0x80 if idx == len(kept_blocks) - 1 else 0x00
        out.append(last_flag | (btype & 0x7F))
        out += len(bdata).to_bytes(3, "big")
        out += bdata
    out += audio_frames
    with open(path, "wb") as f:
        f.write(out)


def stamp_m4a_tags(path, tags):
    """Inserts basic iTunes metadata atoms into afconvert's moov.udta.meta.ilst box in-place,
    shrinking the adjacent padding `free` box so `mdat` offsets and `iTunSMPB` stay untouched."""
    if not tags:
        return
    atom_map = {
        "title": b"\xa9nam",
        "artist": b"\xa9ART",
        "album": b"\xa9alb",
        "date": b"\xa9day",
        "genre": b"\xa9gen",
    }
    extra = b""
    for key, fourcc in atom_map.items():
        val = tags.get(key)
        if not val:
            continue
        payload = str(val).encode("utf-8")
        data_box = struct.pack(">I4sII", 16 + len(payload), b"data", 1, 0) + payload
        atom = struct.pack(">I4s", 8 + len(data_box), fourcc) + data_box
        extra += atom

    if tags.get("tracknumber"):
        parts = str(tags["tracknumber"]).split("/")
        try:
            trk = int(parts[0].strip())
            tot = int(parts[1].strip()) if len(parts) > 1 else 0
            payload = struct.pack(">HHHH", 0, trk, tot, 0)
            data_box = struct.pack(">I4sII", 16 + len(payload), b"data", 0, 0) + payload
            extra += struct.pack(">I4s", 8 + len(data_box), b"trkn") + data_box
        except ValueError:
            pass

    if tags.get("albumversion"):
        mean_b = b"com.apple.iTunes"
        name_b = b"albumversion"
        val_b = str(tags["albumversion"]).encode("utf-8")
        mean_box = struct.pack(">I4sI", 12 + len(mean_b), b"mean", 0) + mean_b
        name_box = struct.pack(">I4sI", 12 + len(name_b), b"name", 0) + name_b
        data_box = struct.pack(">I4sII", 16 + len(val_b), b"data", 1, 0) + val_b
        inner = mean_box + name_box + data_box
        extra += struct.pack(">I4s", 8 + len(inner), b"----") + inner

    if not extra:
        return

    with open(path, "rb") as f:
        buf = bytearray(f.read())

    pos = 0
    moov_pos = None
    while pos + 8 <= len(buf):
        sz, fcc = struct.unpack(">I4s", buf[pos:pos + 8])
        if sz < 8:
            break
        if fcc == b"moov":
            moov_pos = pos
            break
        pos += sz
    if moov_pos is None:
        return

    moov_sz = struct.unpack(">I", buf[moov_pos:moov_pos + 4])[0]
    free_pos = moov_pos + moov_sz
    if free_pos + 8 > len(buf):
        return
    free_sz, free_fcc = struct.unpack(">I4s", buf[free_pos:free_pos + 8])
    if free_fcc != b"free" or free_sz - len(extra) < 8:
        return

    udta_pos = buf.find(b"udta", moov_pos + 8, free_pos) - 4
    meta_pos = buf.find(b"meta", udta_pos + 8, free_pos) - 4
    ilst_pos = buf.find(b"ilst", meta_pos + 12, free_pos) - 4
    if min(udta_pos, meta_pos, ilst_pos) < 0:
        return

    for box_pos in (moov_pos, udta_pos, meta_pos, ilst_pos):
        old_sz = struct.unpack(">I", buf[box_pos:box_pos + 4])[0]
        struct.pack_into(">I", buf, box_pos, old_sz + len(extra))

    new_free_sz = free_sz - len(extra)
    new_free = struct.pack(">I4s", new_free_sz, b"free") + b"\x00" * (new_free_sz - 8)
    buf[free_pos:free_pos + free_sz] = extra + new_free
    with open(path, "wb") as f:
        f.write(buf)


def rewrite_opus_tags(path, comments, pictures=()):
    """Replaces the OpusTags Ogg packet with full Vorbis comments + METADATA_BLOCK_PICTURE cover art."""
    with open(path, "rb") as f:
        buf = f.read()

    pages = []
    pos = 0
    while pos + 27 <= len(buf) and buf[pos:pos + 4] == b"OggS":
        ver, htype, gran, serial, seq, _crc, nsegs = struct.unpack_from("<BBQIIIB", buf, pos + 4)
        header_len = 27 + nsegs
        if pos + header_len > len(buf):
            break
        seg_table = buf[pos + 27:pos + header_len]
        body_start = pos + header_len
        body_end = body_start + sum(seg_table)
        if body_end > len(buf):
            break
        pages.append((ver, htype, gran, serial, seq, buf[body_start:body_end], buf[pos:body_end]))
        pos = body_end

    if len(pages) < 2 or pages[1][5][:8] != b"OpusTags":
        return

    old_tags_body = pages[1][5]
    vlen = struct.unpack_from("<I", old_tags_body, 8)[0]
    vendor = old_tags_body[12:12 + vlen]
    serial = pages[1][3]

    entries = [c.encode("utf-8") for c in comments]
    for pic in pictures:
        entries.append(b"METADATA_BLOCK_PICTURE=" + base64.b64encode(pic))

    pkt = bytearray(b"OpusTags")
    pkt += struct.pack("<I", len(vendor)) + vendor
    pkt += struct.pack("<I", len(entries))
    for entry in entries:
        pkt += struct.pack("<I", len(entry)) + entry

    full_255s, rem = divmod(len(pkt), 255)
    lacing = [255] * full_255s + [rem]

    new_tag_pages = []
    data_offset = 0
    for chunk_idx in range(0, len(lacing), 255):
        seg_chunk = bytes(lacing[chunk_idx:chunk_idx + 255])
        chunk_body_len = sum(seg_chunk)
        chunk_body = pkt[data_offset:data_offset + chunk_body_len]
        data_offset += chunk_body_len
        is_first = chunk_idx == 0
        is_last_chunk = chunk_idx + 255 >= len(lacing)
        htype = 0x00 if is_first else 0x01
        gran = 0 if is_last_chunk else 0xFFFFFFFFFFFFFFFF
        seq = 1 + (chunk_idx // 255)
        hdr = bytearray(struct.pack("<4sBBQIIIB", b"OggS", 0, htype, gran, serial, seq, 0, len(seg_chunk)))
        page_bytes = hdr + seg_chunk + chunk_body
        struct.pack_into("<I", page_bytes, 22, _ogg_crc32(page_bytes))
        new_tag_pages.append(bytes(page_bytes))

    delta_seq = len(new_tag_pages) - 1
    out = bytearray(pages[0][6])
    for tp in new_tag_pages:
        out += tp
    for p in pages[2:]:
        raw_page = bytearray(p[6])
        if delta_seq != 0:
            struct.pack_into("<I", raw_page, 18, p[4] + delta_seq)
            struct.pack_into("<I", raw_page, 22, 0)
            struct.pack_into("<I", raw_page, 22, _ogg_crc32(raw_page))
        out += raw_page

    with open(path, "wb") as f:
        f.write(out)


# ---------------------------------------------------------------------------
# Audio Staging, Measuring & Output Writing
# ---------------------------------------------------------------------------

def write_audio_output(data, rate, dst, fmt, bitrate=None, gain_db=0.0, tags_from=None, include_peaks=True):
    """Applies `gain_db` and writes `data` to `dst` in `fmt` (wav|flac|m4a|aac|opus), returning loudness dict."""
    norm_fmt = "m4a" if fmt == "aac" else fmt
    eff_bitrate = int(bitrate if bitrate is not None else DEFAULT_BITRATES.get(norm_fmt, 256))

    out_audio = data * np.float32(10 ** (gain_db / 20.0)) if gain_db else data.copy()
    np.clip(out_audio, -1.0, 1.0, out=out_audio)
    tags = read_tags(tags_from) if tags_from else {}

    if norm_fmt == "m4a":
        with tempfile.TemporaryDirectory() as tmp:
            gained_wav = os.path.join(tmp, "gained.wav")
            sf.write(gained_wav, out_audio, rate, subtype="PCM_24", format="WAV")
            if sys.platform == "darwin" and os.path.isfile("/usr/bin/afconvert"):
                res = subprocess.run(
                    [
                        "/usr/bin/afconvert",
                        "-f", "m4af",
                        "-d", "aac",
                        "-b", str(eff_bitrate * 1000),
                        "-q", "127",
                        "-s", "2",
                        gained_wav,
                        dst,
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    raise RuntimeError(f"afconvert failed: {res.stderr.strip() or res.stdout.strip()}")
                tags["albumversion"] = "Delimited"
                stamp_m4a_tags(dst, tags)
            else:
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    raise RuntimeError(
                        "AAC (.m4a) output on Linux/Windows requires ffmpeg on PATH "
                        "(or use --format flac / --format opus / --format wav, which require no external tools)"
                    )
                ff_meta = []
                ffmpeg_tag_map = {
                    "title": "title",
                    "artist": "artist",
                    "album": "album",
                    "tracknumber": "track",
                    "date": "date",
                    "genre": "genre",
                }
                for key, ff_key in ffmpeg_tag_map.items():
                    if tags.get(key):
                        ff_meta += ["-metadata", f"{ff_key}={tags[key]}"]
                ff_meta += ["-metadata", "albumversion=Delimited"]
                res = subprocess.run(
                    [
                        ffmpeg, "-nostdin", "-v", "error", "-y",
                        "-i", gained_wav,
                        "-c:a", "aac",
                        "-b:a", f"{eff_bitrate}k",
                        *ff_meta,
                        "-movflags", "+faststart",
                        dst,
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    raise RuntimeError(f"ffmpeg AAC encode failed: {res.stderr.strip()}")

    elif norm_fmt == "opus":
        opus_data = (
            soxr.resample(out_audio, rate, OPUS_RATE, quality="VHQ").astype(np.float32, copy=False)
            if rate != OPUS_RATE else out_audio
        )
        channels = opus_data.shape[1]
        if "OPUS" in sf.available_subtypes("OGG"):
            min_kbps, max_kbps = 6.0 * channels, 256.0 * channels
            vbr_quality = float(np.clip((eff_bitrate - min_kbps) / (max_kbps - min_kbps), 0.0, 1.0))
            with sf.SoundFile(dst, "w", samplerate=OPUS_RATE, channels=channels,
                              format="OGG", subtype="OPUS") as out:
                # SFC_SET_VBR_ENCODING_QUALITY (0x1300) maps [0.0, 1.0] linearly to [6, 256] kbps/channel
                sf._snd.sf_command(out._file, 0x1300, sf._ffi.new("double *", vbr_quality), 8)
                out.write(opus_data)
        else:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("libsndfile lacks Opus support on this system and ffmpeg was not found on PATH")
            with tempfile.TemporaryDirectory() as tmp:
                gained_wav = os.path.join(tmp, "gained48.wav")
                sf.write(gained_wav, opus_data, OPUS_RATE, subtype="FLOAT", format="WAV")
                res = subprocess.run(
                    [
                        ffmpeg, "-nostdin", "-v", "error", "-y",
                        "-i", gained_wav,
                        "-c:a", "libopus",
                        "-b:a", f"{eff_bitrate}k",
                        "-vbr", "on",
                        dst,
                    ],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    raise RuntimeError(f"ffmpeg Opus encode failed: {res.stderr.strip()}")

        flac_comments, flac_pics = extract_flac_tags_and_pictures(tags_from) if tags_from else ([], [])
        if flac_comments or flac_pics:
            opus_comments = flac_comments + ["albumversion=Delimited"]
        else:
            opus_comments = [f"{k.upper()}={v}" for k, v in tags.items() if k.lower() != "albumversion"]
            opus_comments.append("albumversion=Delimited")
        rewrite_opus_tags(dst, opus_comments, flac_pics)

    else:
        file_format = {"wav": "WAV", "flac": "FLAC"}[norm_fmt]
        with sf.SoundFile(dst, "w", samplerate=rate, channels=out_audio.shape[1],
                          subtype="PCM_24", format=file_format) as out:
            for key, value in tags.items():
                setattr(out, key, value)
            out.write(out_audio)
        if norm_fmt == "flac":
            flac_comments, flac_pics = extract_flac_tags_and_pictures(tags_from) if tags_from else ([], [])
            if flac_comments or flac_pics:
                rewrite_flac_metadata(dst, flac_comments + ["albumversion=Delimited"], flac_pics)
            else:
                base_comments = [f"{k.upper()}={v}" for k, v in tags.items() if k.lower() != "albumversion"]
                rewrite_flac_metadata(dst, base_comments + ["albumversion=Delimited"])

    return loudness(out_audio, rate, include_peaks=include_peaks)


def stage(args):
    data, rate = read_audio(args.src)
    data = to_stereo(data)
    if rate != MODEL_RATE:
        data = soxr.resample(data, rate, MODEL_RATE, quality="VHQ").astype(np.float32, copy=False)
    sf.write(args.dst, data, MODEL_RATE, subtype="FLOAT")
    return loudness(data, MODEL_RATE)


def measure(args):
    results = []
    for path in args.files:
        try:
            data, rate = read_audio(path)
            data = to_stereo(data)
            results.append(loudness(data, rate))
        except Exception as error:
            print(f"measure failed for {path}: {error}", file=sys.stderr)
            results.append({"lufs": None, "core_lufs": None, "peak": None, "peaks": []})
    return results


def write(args):
    data, rate = sf.read(args.src, dtype="float32", always_2d=True)
    return write_audio_output(
        data=data,
        rate=rate,
        dst=args.dst,
        fmt=args.format,
        bitrate=args.bitrate,
        gain_db=args.gain,
        tags_from=args.tags_from,
        include_peaks=True,
    )


# ---------------------------------------------------------------------------
# Cross-Platform End-to-End Terminal CLI (`process` / default mode)
# ---------------------------------------------------------------------------

def resolve_delimiter_dir(explicit_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if explicit_dir:
        candidates.append(explicit_dir)
    if os.environ.get("DELIMITER_DIR"):
        candidates.append(os.environ["DELIMITER_DIR"])
    candidates.extend([
        os.path.join(script_dir, "De-limiter"),
        os.path.join(os.getcwd(), "De-limiter"),
        os.getcwd(),
    ])
    for cand in candidates:
        full = os.path.abspath(cand)
        if os.path.isfile(os.path.join(full, "inference.py")):
            return full
    raise FileNotFoundError(
        "Could not find De-limiter/inference.py. Keep the De-limiter folder next to delimiter_tool.py "
        "or pass --delimiter-dir /path/to/De-limiter."
    )


def collect_cli_tracks(inputs, custom_output_dir):
    """Discovers audio files across `inputs` (files or folders) and resolves each track's output directory."""
    dir_inputs = [os.path.abspath(p) for p in inputs if os.path.isdir(p)]
    multi_dir_inputs = len(dir_inputs) > 1
    entries = []
    seen = set()

    for raw_inp in inputs:
        inp = os.path.abspath(raw_inp)
        if not os.path.exists(inp):
            print(f"WARNING: input path does not exist, skipping: {raw_inp}", file=sys.stderr)
            continue

        if os.path.isfile(inp):
            ext = os.path.splitext(inp)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"WARNING: unsupported file extension, skipping: {raw_inp}", file=sys.stderr)
                continue
            norm_key = os.path.normcase(inp)
            if norm_key in seen:
                continue
            seen.add(norm_key)
            parent = os.path.dirname(inp)
            dest_dir = os.path.abspath(custom_output_dir) if custom_output_dir else (parent + "_delimited")
            entries.append({"src": inp, "album_key": parent, "dest_dir": dest_dir})
            continue

        # Directory input: walk recursively in sorted order
        found_files = []
        for dirpath, dirnames, filenames in os.walk(inp):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                    found_files.append(os.path.join(dirpath, fname))

        for fpath in found_files:
            norm_key = os.path.normcase(fpath)
            if norm_key in seen:
                continue
            seen.add(norm_key)
            album_dir = os.path.dirname(fpath)
            rel_sub = os.path.relpath(album_dir, inp)
            if custom_output_dir:
                base_out = os.path.abspath(custom_output_dir)
                if multi_dir_inputs:
                    base_out = os.path.join(base_out, os.path.basename(inp.rstrip("/\\")))
            else:
                base_out = inp.rstrip("/\\") + "_delimited"
            dest_dir = base_out if rel_sub == "." else os.path.join(base_out, rel_sub)
            entries.append({"src": fpath, "album_key": album_dir, "dest_dir": dest_dir})

    return entries


def unique_stem(stem, dest_dir, used_paths):
    candidate = stem
    counter = 2
    while os.path.normcase(os.path.join(dest_dir, candidate)) in used_paths:
        candidate = f"{stem} ({counter})"
        counter += 1
    used_paths.add(os.path.normcase(os.path.join(dest_dir, candidate)))
    return candidate


def copy_album_sidecars(album_in_dir, album_out_dir):
    if not os.path.isdir(album_in_dir):
        return
    for fname in sorted(os.listdir(album_in_dir)):
        if fname.startswith("."):
            continue
        src = os.path.join(album_in_dir, fname)
        if not os.path.isfile(src):
            continue
        if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
            dst = os.path.join(album_out_dir, fname)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)


def select_torch_device(requested_device, torch_mod):
    req = requested_device.lower()
    if req == "auto":
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch_mod.device("mps")
        return torch_mod.device("cpu")
    if req in ("cuda", "gpu"):
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        print("WARNING: CUDA is not available in this PyTorch environment; falling back to CPU.", file=sys.stderr)
        return torch_mod.device("cpu")
    if req == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return torch_mod.device("mps")
        print("WARNING: Apple Silicon MPS is not available; falling back to CPU.", file=sys.stderr)
        return torch_mod.device("cpu")
    return torch_mod.device("cpu")


def process_cli(args):
    delimiter_dir = resolve_delimiter_dir(args.delimiter_dir)
    weight_dir = os.path.join(delimiter_dir, "weight")
    cfg_path = os.path.join(weight_dir, f"{args.target}.json")
    ckpt_path = os.path.join(weight_dir, f"{args.target}.pth")
    if not os.path.isfile(cfg_path) or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Model '{args.target}' not found in {weight_dir} (expected {args.target}.json and {args.target}.pth)"
        )

    entries = collect_cli_tracks(args.inputs, args.output_dir)
    if not entries:
        raise RuntimeError("No supported audio files found in the specified input path(s).")

    # Group tracks by source album folder (preserves discovery order)
    albums = OrderedDict()
    for entry in entries:
        albums.setdefault(entry["album_key"], []).append(entry)

    import torch
    import tqdm
    if delimiter_dir not in sys.path:
        sys.path.insert(0, delimiter_dir)
    from dotmap import DotMap
    from models import load_model_with_args
    from separate_func import conv_tasnet_separate
    from utils import db2linear

    device = select_torch_device(args.device, torch)
    out_ext = FORMAT_EXTENSIONS[args.format]
    eff_bitrate = int(args.bitrate if args.bitrate is not None else DEFAULT_BITRATES.get(args.format, 256))

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_dict = DotMap(json.load(f))

    if args.loudness_mode == "track":
        save_output_loudnorm = float(args.target_lufs)
    elif args.loudness_mode == "album":
        save_output_loudnorm = -14.0
    else:
        save_output_loudnorm = None

    model_args = argparse.Namespace(
        target=args.target,
        weight_directory=weight_dir,
        use_gpu=device.type == "cuda",
        use_mps=device.type == "mps",
        save_name_as_target=False,
        loudnorm_input_lufs=args.input_loudnorm,
        save_output_loudnorm=save_output_loudnorm,
        save_mixed_output=None,
        save_16k_mono=False,
        save_histogram=False,
        use_singletrackset=args.chunk_long_tracks,
        save_sgi_params=False,
        avoid_clipping=args.avoid_clipping,
    )
    for key, value in cfg_dict["args"].items():
        if not hasattr(model_args, key):
            setattr(model_args, key, value)

    fmt_desc = out_ext.upper()
    if out_ext in ("m4a", "opus"):
        fmt_desc += f" ({eff_bitrate} kbps)"
    print(
        f"De-limiter Studio CLI | Device: {device.type.upper()} | Model: {args.target} | "
        f"Mode: {args.loudness_mode} | Output: {fmt_desc}",
        flush=True,
    )
    print(f"Found {len(entries)} track(s) across {len(albums)} album/folder group(s).\n", flush=True)

    model = load_model_with_args(model_args).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    meter = pyloudnorm.Meter(MODEL_RATE)

    used_output_paths = set()
    total_succeeded = 0
    t_batch_start = time.time()

    for album_idx, (album_dir, track_list) in enumerate(albums.items(), 1):
        album_label = os.path.basename(album_dir.rstrip("/\\")) or album_dir
        print(f"=== [{album_idx}/{len(albums)}] {album_label} ({len(track_list)} track(s)) ===", flush=True)
        t_album = time.time()

        with tempfile.TemporaryDirectory(prefix="delimiter_cli_") as tmp_dir:
            model_args.test_output_dir = tmp_dir
            staged = []

            print("  1/3 Staging & measuring source audio...", flush=True)
            for t_idx, item in enumerate(track_list, 1):
                src_path = item["src"]
                fname = os.path.basename(src_path)
                try:
                    data, rate = read_audio(src_path)
                    data = to_stereo(data)
                    if rate != MODEL_RATE:
                        data = soxr.resample(data, rate, MODEL_RATE, quality="VHQ").astype(np.float32, copy=False)
                    orig_m = loudness(data, MODEL_RATE, include_peaks=False)
                except Exception as exc:
                    print(f"      [ERROR] {fname}: {exc}", file=sys.stderr, flush=True)
                    continue

                l_str = f"{orig_m['lufs']:.2f} LUFS" if orig_m["lufs"] is not None else "N/A"
                p_str = f"{orig_m['peak']:.2f} dBTP" if orig_m["peak"] is not None else "N/A"
                print(f"      - {fname}: orig = {l_str}, peak = {p_str}", flush=True)
                staged.append({
                    "entry": item,
                    "fname": fname,
                    "stem": os.path.splitext(fname)[0],
                    "token": f"track_{t_idx:04d}",
                    "audio": data,
                    "orig_lufs": orig_m["lufs"],
                    "orig_peak": orig_m["peak"],
                })

            if not staged:
                continue

            print(f"  2/3 Running De-limiter inference ({device.type.upper()})...", flush=True)
            infer_ok = []
            for st in tqdm.tqdm(staged, desc="      De-limiting", unit="trk"):
                track_audio = np.ascontiguousarray(st["audio"].T)
                st["audio"] = None  # release staged float array once transposed
                augmented_gain = None
                if model_args.loudnorm_input_lufs is not None:
                    try:
                        in_lufs = meter.integrated_loudness(track_audio.T)
                        if math.isfinite(in_lufs):
                            augmented_gain = model_args.loudnorm_input_lufs - in_lufs
                            track_audio = track_audio * db2linear(augmented_gain, eps=0.0)
                    except ValueError:
                        pass

                tensor_in = torch.as_tensor(track_audio, dtype=torch.float32).unsqueeze(0).to(device)
                try:
                    with torch.no_grad():
                        conv_tasnet_separate(
                            model_args, model, device, tensor_in, st["token"],
                            meter=meter, augmented_gain=augmented_gain,
                        )
                except RuntimeError as err:
                    if not model_args.use_singletrackset:
                        print(f"\n      Retrying '{st['fname']}' in chunked mode (--chunk-long-tracks)...", flush=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps" and hasattr(torch, "mps"):
                            torch.mps.empty_cache()
                        model_args.use_singletrackset = True
                        try:
                            with torch.no_grad():
                                conv_tasnet_separate(
                                    model_args, model, device, tensor_in, st["token"],
                                    meter=meter, augmented_gain=augmented_gain,
                                )
                        finally:
                            model_args.use_singletrackset = args.chunk_long_tracks
                    else:
                        print(f"\n      [ERROR] Inference failed on {st['fname']}: {err}", file=sys.stderr, flush=True)
                        continue
                finally:
                    del tensor_in

                raw_wav = os.path.join(tmp_dir, f"{st['token']}.wav")
                if not os.path.isfile(raw_wav):
                    print(f"\n      [ERROR] Missing raw output for {st['fname']}", file=sys.stderr, flush=True)
                    continue
                raw_data, raw_rate = sf.read(raw_wav, dtype="float32", always_2d=True)
                raw_m = loudness(raw_data, raw_rate, include_peaks=False)
                st["raw_wav"] = raw_wav
                st["raw_lufs"] = raw_m["lufs"]
                st["raw_peak"] = raw_m["peak"]
                infer_ok.append(st)

            if not infer_ok:
                continue

            if args.loudness_mode == "album":
                valid_orig = [s["orig_lufs"] for s in infer_ok if s["orig_lufs"] is not None]
                loudest_orig = max(valid_orig) if valid_orig else args.target_lufs
                shift = 0.0
                for s in infer_ok:
                    if s["orig_lufs"] is not None and s["raw_lufs"] is not None and s["raw_peak"] is not None:
                        gain = args.target_lufs + (s["orig_lufs"] - loudest_orig) - s["raw_lufs"]
                        s["gain_db"] = gain
                        shift = min(shift, args.peak_ceiling - s["raw_peak"] - gain)
                    else:
                        s["gain_db"] = 0.0
                for s in infer_ok:
                    s["gain_db"] += shift
                print(
                    f"      Album rebalance: loudest orig = {loudest_orig:.2f} LUFS, "
                    f"ceiling shift = {shift:+.2f} dB",
                    flush=True,
                )
            else:
                for s in infer_ok:
                    s["gain_db"] = 0.0

            print(f"  3/3 Writing {fmt_desc} output...", flush=True)
            copied_sidecar_dirs = set()
            for s in infer_ok:
                dest_dir = s["entry"]["dest_dir"]
                os.makedirs(dest_dir, exist_ok=True)
                base = unique_stem(s["stem"], dest_dir, used_output_paths)
                dst_path = os.path.join(dest_dir, f"{base}.{out_ext}")
                raw_data, raw_rate = sf.read(s["raw_wav"], dtype="float32", always_2d=True)
                out_m = write_audio_output(
                    data=raw_data,
                    rate=raw_rate,
                    dst=dst_path,
                    fmt=args.format,
                    bitrate=eff_bitrate,
                    gain_db=s["gain_db"],
                    tags_from=s["entry"]["src"] if args.copy_tags else None,
                    include_peaks=False,
                )
                fl_str = f"{out_m['lufs']:.2f} LUFS" if out_m["lufs"] is not None else "N/A"
                fp_str = f"{out_m['peak']:.2f} dBTP" if out_m["peak"] is not None else "N/A"
                print(
                    f"      -> {os.path.basename(dst_path)} "
                    f"(gain {s['gain_db']:+.2f} dB | final {fl_str}, peak {fp_str})",
                    flush=True,
                )
                total_succeeded += 1
                if args.copy_sidecars and dest_dir not in copied_sidecar_dirs:
                    copy_album_sidecars(album_dir, dest_dir)
                    copied_sidecar_dirs.add(dest_dir)

        print(f"  Finished '{album_label}' in {time.time() - t_album:.1f}s.\n", flush=True)

    elapsed = time.time() - t_batch_start
    print(f"Done: {total_succeeded}/{len(entries)} track(s) processed in {elapsed:.1f}s.", flush=True)
    if total_succeeded < len(entries):
        sys.exit(1)


def build_cli_parser():
    parser = argparse.ArgumentParser(
        prog="delimiter_tool.py",
        description=(
            "De-limiter Studio CLI (macOS, Linux, Windows): restore dynamic range and transient headroom "
            "to brickwall-limited audio with the same defaults and album-relative rebalancing as the GUI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python delimiter_tool.py ./MyAlbum\n"
            "  python delimiter_tool.py ./MyLibrary --device auto --format opus --bitrate 128\n"
            "  python delimiter_tool.py song1.flac song2.flac -d cuda -f flac -l -13.0 -c -1.0 -o ./Out\n"
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="One or more audio files (.flac, .wav, .mp3, .m4a, .aiff, .ogg, .opus) or folders",
    )
    parser.add_argument(
        "-f", "--format",
        choices=("flac", "wav", "aac", "m4a", "opus"),
        default="flac",
        help="Output audio format (default: flac, matching GUI)",
    )
    parser.add_argument(
        "-b", "--bitrate",
        type=int,
        default=None,
        help="Target VBR bitrate in kbps for AAC/Opus (default: 256 for AAC, 192 for Opus)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        metavar="DIR",
        help="Custom output directory (default: <Input>_delimited next to each input)",
    )
    parser.add_argument(
        "-d", "--device",
        choices=("cpu", "cuda", "gpu", "mps", "auto"),
        default="cpu",
        help="Compute device: cpu (default), cuda/gpu (NVIDIA/ROCm), mps (Apple Silicon), or auto",
    )
    parser.add_argument(
        "-m", "--loudness-mode",
        choices=("album", "track", "raw"),
        default="album",
        help="Loudness strategy: album (relative rebalance, default), track (per-track loudnorm), or raw",
    )
    parser.add_argument(
        "-l", "--target-lufs",
        type=float,
        default=-14.0,
        metavar="LUFS",
        help="Target integrated loudness in LUFS for 'album' (loudest track) or 'track' mode (default: -14.0)",
    )
    parser.add_argument(
        "-c", "--peak-ceiling",
        type=float,
        default=-1.0,
        metavar="DBTP",
        help="True Peak ceiling in dBTP for 'album' relative rebalance mode (default: -1.0)",
    )
    parser.add_argument(
        "--input-loudnorm",
        type=float,
        default=None,
        metavar="LUFS",
        help="Optional pre-inference input loudness target in LUFS (e.g. -14.0; default: disabled)",
    )
    parser.add_argument(
        "--chunk-long-tracks",
        action="store_true",
        default=False,
        help="Process tracks in overlapping chunks (--use_singletrackset) to lower memory/VRAM usage",
    )
    parser.add_argument(
        "--avoid-clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize raw model output if any sample exceeds 0 dBFS (default: --avoid-clipping)",
    )
    parser.add_argument(
        "--copy-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy source metadata tags & cover art and stamp albumversion=Delimited (default: --copy-tags)",
    )
    parser.add_argument(
        "--copy-sidecars",
        action="store_true",
        default=False,
        help="Copy non-audio sidecar files (.lrc, cover.jpg, etc.) when processing folders",
    )
    parser.add_argument(
        "--target",
        default="all",
        metavar="MODEL",
        help="Pretrained model name inside De-limiter/weight/ (default: all)",
    )
    parser.add_argument(
        "--delimiter-dir",
        default=None,
        metavar="DIR",
        help="Path to the De-limiter directory containing inference.py and weight/ (default: auto)",
    )
    return parser


def build_internal_parser():
    parser = argparse.ArgumentParser(description="Internal GUI helper subcommands for De-limiter.app")
    commands = parser.add_subparsers(dest="command", required=True)

    stage_cmd = commands.add_parser("stage", help="decode to a 44.1 kHz stereo float WAV and measure it")
    stage_cmd.add_argument("src")
    stage_cmd.add_argument("dst")
    stage_cmd.set_defaults(run=stage)

    measure_cmd = commands.add_parser("measure", help="measure loudness and true peak")
    measure_cmd.add_argument("files", nargs="+")
    measure_cmd.set_defaults(run=measure)

    write_cmd = commands.add_parser("write", help="apply gain and write 24-bit WAV/FLAC, AAC (.m4a), or Opus (.opus)")
    write_cmd.add_argument("src")
    write_cmd.add_argument("dst")
    write_cmd.add_argument("--format", choices=("wav", "flac", "m4a", "aac", "opus"), required=True)
    write_cmd.add_argument("--bitrate", type=int, default=None, help="AAC/Opus bitrate in kbps")
    write_cmd.add_argument("--gain", type=float, default=0.0, help="gain in dB")
    write_cmd.add_argument("--tags-from", help="copy metadata tags from this file")
    write_cmd.set_defaults(run=write)
    return parser


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("stage", "measure", "write"):
        parser = build_internal_parser()
        args = parser.parse_args()
        try:
            result = args.run(args)
        except Exception as error:
            sys.exit(f"{args.command} failed: {error}")
        print("RESULT " + json.dumps(result), flush=True)
        return

    cli_argv = sys.argv[1:]
    if cli_argv and cli_argv[0] == "process":
        cli_argv = cli_argv[1:]
    parser = build_cli_parser()
    args = parser.parse_args(cli_argv)
    try:
        process_cli(args)
    except KeyboardInterrupt:
        sys.exit("\nCancelled by user.")
    except Exception as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()

