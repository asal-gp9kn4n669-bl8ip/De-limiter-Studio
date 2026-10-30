# De-limiter Studio

A native **macOS (SwiftUI) application** and **cross-platform command-line tool (Linux, Windows, macOS)** for restoring dynamic range and transient headroom to brickwall-limited audio, built on top of the [**De-limiter**](https://github.com/jeonchangbin49/De-limiter) Sample-wise Gain Inversion (SGI) network (*WASPAA 2023*).

---

## Features

- **Album / Batch Relative Loudness Rebalancing**: Automatically groups tracks by album folder, anchors the loudest track in each album to your target LUFS (`-14.0 LUFS` default), preserves exact track-to-track relative volume differences, and shifts the album down together if any restored transient exceeds your True Peak Ceiling (`-1.0 dBTP` default).
- **Core-Volume-Matched Waveform Visualization *(macOS GUI)***: Overlays the original waveform on top of the de-limited waveform matched by sustained **Core Loudness** (ITU-R BS.1770-4 K-weighted 10th–60th percentile energy), visually highlighting restored peaks rising above the original brickwall ceiling and reporting restored peak headroom (`+dB`).
- **Synchronized Gapless A/B Player *(macOS GUI)***: Instant switching between **A: Original** and **B: De-limited** with **Relative Volume Match** so you hear pure transient restoration without loudness bias.
- **Lossless, Opus & AAC Output (with Metadata & Cover Art Preservation)**:
  - **FLAC** (24-bit / 44.1 kHz lossless) — preserves source Vorbis comments, embedded cover art, and stamps `albumversion=Delimited`.
  - **WAV** (24-bit / 44.1 kHz lossless) — preserves basic RIFF INFO tags.
  - **Opus (`.opus`)** — VBR (`64–320 kbps`, `192 kbps` default), resampled to 48 kHz with SoXR VHQ via bundled `libopus` (`libsndfile`), preserving source Vorbis comments, embedded cover art (`METADATA_BLOCK_PICTURE`), and `albumversion=Delimited`.
  - **AAC (`.m4a`)** — `256 kbps` default (`96–320 kbps`), encoded via Apple Core Audio `afconvert` Constrained VBR (`-q 127 -s 2`) on macOS or `ffmpeg` on Linux/Windows, with tag preservation and `albumversion=Delimited`.
- **Cross-Platform Hardware Acceleration**: Supports **CPU**, **NVIDIA / ROCm GPU (`cuda`)** on Linux & Windows, and **Apple Silicon GPU (`mps`)** on macOS.

---

## macOS App (Pre-Built Release)

The pre-built macOS release (`De-limiter-Studio-v1.1.0-macOS-arm64.zip`) is **100% self-contained**—it bundles a standalone Python 3.11 runtime, PyTorch, `libsndfile` (`libFLAC` + `libopus`), audio libraries, and the pretrained model weights inside `De-limiter.app`. No Python, Homebrew, or `ffmpeg` installation is required.

1. Download **`De-limiter-Studio-v1.1.0-macOS-arm64.zip`** from the [**Releases**](../../releases) page.
2. Extract the archive and drag **`De-limiter.app`** to `/Applications` (or anywhere you like).
3. Because the app is not notarized with an Apple Developer ID, run this once in Terminal to clear the macOS quarantine flag:
   ```bash
   xattr -dr com.apple.quarantine /Applications/De-limiter.app
   ```
4. Launch **`De-limiter.app`**.

*macOS GUI Requirements: Apple Silicon Mac (M1 / M2 / M3 / M4) running macOS 14.0 (Sonoma) or later.*

---

## Running from the Terminal (Linux, Windows & macOS)

`delimiter_tool.py` provides a full cross-platform CLI with the **exact same processing pipeline, loudness rebalancing, format conversion options, and defaults** as the macOS GUI.

### 1. CLI Setup

Requires **Python 3.10+** (3.11 recommended). Clone this repository and install the runtime dependencies inside a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-app.in
```

> **Notes for Linux & Windows Users:**
> - **NVIDIA GPU Acceleration (`--device cuda`)**: Install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before (or instead of) the default PyPI `torch` wheel if needed on Windows/Linux.
> - **External Tools**: **FLAC**, **WAV**, and **Opus** require **no external tools** on Windows, Linux, or macOS (handled directly by `soundfile` / `libsndfile` + `soxr`). Only `.m4a` (AAC) encoding/decoding on Linux and Windows requires `ffmpeg` on `PATH`.

*(Tip for macOS `De-limiter.app` users: you can also invoke the bundled runtime directly without creating a virtualenv via `/Applications/De-limiter.app/Contents/Resources/python/bin/python3 delimiter_tool.py ...`)*

### 2. CLI Usage Examples

```bash
# Process an album folder using GUI defaults:
# CPU, Album Relative Rebalance (-14.0 LUFS / -1.0 dBTP), 24-bit FLAC -> ./MyAlbum_delimited/
python delimiter_tool.py ./MyAlbum

# Process a multi-album music folder on GPU (CUDA on Linux/Windows, MPS on macOS) to 128 kbps Opus,
# copying .lrc lyrics and cover.jpg sidecar files into ./MyLibrary_delimited/<Album>/
python delimiter_tool.py ./MyLibrary --device auto --format opus --bitrate 128 --copy-sidecars

# Process on NVIDIA GPU (Linux/Windows) to 24-bit FLAC with a -12.0 LUFS album target
python delimiter_tool.py ./MyAlbum -d cuda -f flac -l -12.0 -c -1.0

# Process individual tracks to 256 kbps AAC (.m4a) into a custom output directory
python delimiter_tool.py track1.wav track2.wav -f aac -b 256 -o ./Delimited_AAC
```

### 3. CLI Options & Flags

All defaults match the **De-limiter Studio** GUI:

| Option / Flag | Values / Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `INPUT [INPUT ...]` | File(s) or Folder(s) | *(Required)* | Audio files (`.flac`, `.wav`, `.mp3`, `.m4a`, `.aiff`, `.ogg`, `.opus`) or directories (scanned recursively and grouped by album folder). |
| `-f`, `--format` | `flac`, `wav`, `aac` (`m4a`), `opus` | `flac` | Output format: 24-bit/44.1 kHz `flac` or `wav`, `aac` (`.m4a`), or 48 kHz VBR `opus` (`.opus`). |
| `-b`, `--bitrate` | `INT` (kbps) | `256` (`aac`), `192` (`opus`) | Target VBR bitrate in kbps for lossy output (`96–320` typical for AAC; `64–320` typical for Opus). Ignored for `flac` and `wav`. |
| `-o`, `--output-dir` | `DIR` | `<Input>_delimited` | Custom output directory. When omitted, writes to `<Input>_delimited` alongside each input (preserving album subfolders). |
| `-d`, `--device` | `cpu`, `cuda` (`gpu`), `mps`, `auto` | `cpu` | Inference device: `cpu` (GUI default), `cuda` / `gpu` (NVIDIA/ROCm on Linux & Windows), `mps` (Apple Silicon GPU), or `auto` (best available GPU, else CPU). |
| `-m`, `--loudness-mode` | `album`, `track`, `raw` | `album` | Loudness strategy: `album` (Album/Batch Relative Rebalance), `track` (per-track loudnorm), or `raw` (unnormalized model output). |
| `-l`, `--target-lufs` | `FLOAT` (LUFS) | `-14.0` | Target integrated loudness in LUFS (anchors the loudest track per album in `album` mode, or every track in `track` mode). |
| `-c`, `--peak-ceiling` | `FLOAT` (dBTP) | `-1.0` | Maximum 4x-oversampled True Peak ceiling in dBTP (`album` mode). Shifts the whole album down together if any restored peak exceeds ceiling. |
| `--input-loudnorm` | `FLOAT` (LUFS) | Disabled | Optional pre-inference input loudness normalization (`--loudnorm_input_lufs`), e.g. `-14.0`. |
| `--chunk-long-tracks` | Flag | Off | Process tracks in overlapping windows (`--use_singletrackset`) to lower RAM/VRAM usage (also triggered automatically on GPU OOM). |
| `--avoid-clipping` / `--no-avoid-clipping` | Boolean flag | `--avoid-clipping` | Forcefully normalize raw model output if any sample exceeds `0 dBFS`. |
| `--copy-tags` / `--no-copy-tags` | Boolean flag | `--copy-tags` | Copy source metadata tags (plus embedded FLAC cover art for FLAC/Opus) and stamp `albumversion=Delimited` on FLAC, AAC, and Opus outputs. |
| `--copy-sidecars` | Flag | Off | Copy non-audio sidecar files (e.g., `.lrc` lyrics, `cover.jpg`) into each output album folder when processing directories. |
| `--target` | `MODEL` | `all` | Pretrained model checkpoint name inside `De-limiter/weight/` (`<name>.pth` + `<name>.json`). |
| `--delimiter-dir` | `DIR` | `./De-limiter` | Custom path to the `De-limiter` folder (can also be set via the `DELIMITER_DIR` environment variable). |

---

## Building the macOS App from Source

### Prerequisites
- **macOS 14.0+** on Apple Silicon (`arm64`)
- **Xcode Command Line Tools** (`xcode-select --install`) for `swiftc`
- [**uv**](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`) to provision the standalone Python 3.11 runtime

### Build Commands
```bash
# Full build (provisions standalone Python 3.11 + packages into De-limiter.app and compiles SwiftUI app):
./build_app.sh --full

# Fast incremental build (recompiles DelimiterApp.swift and syncs delimiter_tool.py + De-limiter/):
./build_app.sh

# Create a distributable zip archive (preserves bundle symlinks via ditto):
./build_app.sh --zip
```

---

## Repository Structure

```text
├── DelimiterApp.swift      # Native macOS SwiftUI interface, A/B player & batch pipeline
├── delimiter_tool.py       # Cross-platform CLI (Linux/Windows/macOS), loudness engine & WAV/FLAC/AAC/Opus writer
├── build_app.sh            # Self-contained macOS De-limiter.app builder and packager
├── requirements-app.in     # Cross-platform Python dependencies (CLI + app runtime)
├── requirements-app.txt    # Pinned macOS arm64 lockfile for reproducible De-limiter.app builds
└── De-limiter/             # Streamlined inference-only adaptation of jeonchangbin49/De-limiter + weights (all.pth)
```

---

## Attribution & Citation

De-limiter Studio is a native macOS GUI and cross-platform CLI built around an inference-only adaptation of [**`jeonchangbin49/De-limiter`**](https://github.com/jeonchangbin49/De-limiter) (updated in `De-limiter/` for PyTorch 2.x, Apple Silicon MPS acceleration, and 32-bit float headroom preservation).

All credit for the underlying **De-limiter** neural network architecture, Sample-wise Gain Inversion (SGI) methodology, and pretrained model weights (`De-limiter/weight/all.pth`) belongs to **Chang-Bin Jeon** and **Kyogu Lee**:

- **Original Repository**: [https://github.com/jeonchangbin49/De-limiter](https://github.com/jeonchangbin49/De-limiter)
- **Paper**: *"Music De-Limiter Networks via Sample-wise Gain Inversion"*, IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2023 ([arXiv:2308.01187](https://arxiv.org/abs/2308.01187)).

```bibtex
@inproceedings{jeon2023music,
  title={Music De-Limiter Networks via Sample-wise Gain Inversion},
  author={Jeon, Chang-Bin and Lee, Kyogu},
  booktitle={2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023},
  organization={IEEE}
}
```

The `Conv-TasNet` filterbank and temporal convolutional masking layers in `De-limiter/models/` are adapted from [**Asteroid**](https://github.com/asteroid-team/asteroid) (MIT License, *Pariente et al., Interspeech 2020*).

---

## License

Released under the [MIT License](LICENSE). See [LICENSE](LICENSE) for full copyright and third-party license notices.
