# diarisation-benchmark
First steps in setting up a diarisation benchmark for Slovenian and related languages

## Dataset

We will start with the open dataset ROG-Dialog http://hdl.handle.net/11356/2073. The audio is to be taken from the repository, while the rttm format is available in this repository for simplicity (original repository contains XML Exmaralda files that can be investigated if needed, editor is this: https://exmaralda.org/en/).

## Models

Models to be evaluated in the first iteration are
- pyannote (legacy 3.1, community-1, precision-2, or any others looking promising) https://huggingface.co/pyannote
- NVIDIA softformer https://huggingface.co/nvidia/diar_sortformer_4spk-v1
- NVIDIA NeMo models?
- SpeechBrain models?
- any other models identified as promising
- feel free to spend a few EUR (and bill us for these) on API-based diarisers (precision-2 etc.), if they perform significantly better, we are happy to use these as well for some data

## Evaluation

While all model outputs are to be logged for future evaluation runs, the first iteration should report
- diarisation error rate (DER) pyannote.metrics.diarization.DiarizationErrorRate
- processing speed

We are very open to additional metrics as well.



# Peter : Running the pipeline through and through

1. Download the data:

`bash prepare_data.sh`

2. Run pyannote:
    1. Build the docker image:

    ```bash
    cd models/pyannote
    docker build -t benchmark-pyannote .
    cd ../..
    ```

    2. Run first model:

    ```bash
    sudo docker run --rm \
        -v "$(pwd)/data/ROG-Dialog/audio:/data/audio" \
        -v "$(pwd)/results/pyannote_3_1:/data/output" \
        -e HOST_UID=$(id -u) \
        -e HOST_GID=$(id -g) \
        -e HF_TOKEN="YOURTOKEN" \
        benchmark-pyannote \
        --input /data/audio \
        --output /data/output \
        --model pyannote/speaker-diarization-3.1
    ```

    This works, but on GPU2 there is no docker, and on my laptop, there is no GPU.

    Consequently, processing takes ages, with RTF = 2!


3. Nemo models:
   1. Build the docker image:

    ```bash
    cd models/nemo
    sudo docker build -t benchmark-nemo .
    cd ../..
    ```

    2. Run first model:

    ```bash
    sudo docker run --rm \
        -v "$(pwd)/data/ROG-Dialog/audio:/data/audio" \
        -v "$(pwd)/results/nemo_v2:/data/output" \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -e HOST_UID=$(id -u) \
        -e HOST_GID=$(id -g) \
        -e HF_TOKEN="YOURTOKEN" \
        benchmark-nemo \
        --input /data/audio \
        --output /data/output
    ```

    This runs faster, with RTF of 0.1 cca.

4. Run the eval

```bash
cd evaluation
sudo docker build -t benchmark-eval .
cd ..

sudo docker run --rm \
  -v "$(pwd)/data/ROG-Dialog:/data/rog" \
  -v "$(pwd)/results:/data/results" \
  -v "$(pwd)/reports:/data/reports" \
  -v "$(pwd)/evaluation/DATASET_ERRATA.json:/app/DATASET_ERRATA.json" \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  benchmark-eval \
  --gold /data/rog/ref_rttm/gold_standard.rttm \
  --results_dir /data/results \
  --metadata /data/rog/docs/ROG-Dia-meta-speeches.tsv \
  --output /data/reports/ROG-Dia_Final_Report
```

# Ivan: Auto-trim silences from Gold intervals with Praat (Apr2026)

Human-annotated segment boundaries in the gold RTTM often include leading/trailing silence (annotators clicking a bit too early or too late). The `trim_gold_silences_rttm.py` module uses Praat's pitch and intensity analysis (via Parselmouth) to detect actual speech onset/offset and tighten those boundaries automatically. It can also split segments at long internal silences.

## What it does

- Loads each audio file and analyses segments using pitch detection + intensity relative to segment peak
- Trims segment edges to where speech actually starts/ends, with a configurable guard margin
- Optionally splits segments at internal silences (e.g. ≥500ms gaps within a single annotation)
- Drops segments that become too short after trimming (configurable threshold)
- Writes results incrementally per file (crash-safe — no data lost if it fails mid-run)

### Impact of Gold Standard Trimming on Evaluation Metrics

Trimming the gold standard consistently lowers DER across all models, driven almost entirely by reduced Miss rates — the original annotations include silence at segment edges that unfairly penalizes models for not predicting speech where there is none. FA increases slightly (smaller speech denominator), while Confusion stays stable since trimming doesn't affect speaker identity.

Example using the best-performing model (pyannote speaker-diarization-precision-2, collar=0.25):

| Metric   | Original Gold | Trimmed Gold |
|----------|---------------|--------------|
| DER      | 20.25%        | **9.52%**    |
| Miss     | 17.40%        | **5.78%**    |
| FA       | **1.26%**     | 2.37%        |
| Conf     | **1.22%**     | 1.36%        |
| Purity   | **86.91%**    | 86.89%       |
| Coverage | **89.32%**    | 89.09%       |

The trimmed evaluation better reflects actual diarisation performance by removing measurement artifacts from imprecise annotation boundaries.

## Two ways to use it

### 1. Standalone CLI (trim an existing RTTM)

```bash
python trim_gold_silences_rttm.py \
    --rttm data/ROG-Dialog/ref_rttm/gold_standard.rttm \
    --audio-dir data/ROG-Dialog/audio \
    --output data/ROG-Dialog/ref_rttm/gold_trimmed.rttm \
    --trim-silence-within \
    --verbose
```

Run `python trim_gold_silences_rttm.py --help` for all options (pitch range, intensity threshold, guard margin, max trim, etc.).

### 2. Integrated in the TRS→RTTM pipeline (recommended)

`convert_trs_to_trim_rttm.py` imports the trimming module and runs the full pipeline: parse TRS → merge segments → trim with audio → write RTTM + optional EXB files. All settings are configured at the top of the script:

```python
ENABLE_TRIMMING = True       # set False to skip audio analysis
GENERATE_EXB = True          # generate EXB files for visual inspection
TRIM_PARAMS = TrimParams(
    intensity_drop_db=15.0,  # dB below segment peak = "silence"
    trim_silence_within=True,
    min_silence_dur=0.5,
    verbose=False,
    # ... other params with sensible defaults
)
```

```bash
python convert_trs_to_trim_rttm.py
```

Output filename is automatic: `gold_standard_trimmed_{int}db.rttm` when trimming is on, `gold_standard.rttm` when off. A `_metadata.txt` file with full parameters and statistics is written alongside.

### EXB output

When `GENERATE_EXB = True`, the script produces EXB files with `[Dia_gold_trim]` tiers that can be opened in EXMARaLDA Partitur Editor alongside the original transcription tiers for visual verification of trim quality.

### Note on file_id

TRS filenames (e.g. `ROG-Dia-GSO-P0005-std.trs`) are stripped of `-std`/`-pog` suffixes to derive the file ID (`ROG-Dia-GSO-P0005`). This must match the corresponding `.wav` and `.exb` filenames.