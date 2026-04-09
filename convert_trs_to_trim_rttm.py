#!/usr/bin/env python3
"""
trs_to_gold_rttm.py — Convert TRS transcription files to a gold-standard RTTM,
with optional audio-informed silence trimming and EXB generation.

Pipeline:
  1. Parse TRS files (one per recording, STD or POG variant)
  2. Merge adjacent same-speaker segments (linear merge)
  3. Optionally trim segment boundaries using audio analysis (Praat/Parselmouth)
  4. Write RTTM incrementally per file (crash-safe)
  5. Optionally generate EXB files for visual inspection

NOTE on file_id derivation:
  TRS filenames like "ROG001-std.trs" or "ROG001-pog.trs" are stripped to "ROG001".
  This must match the .wav filenames in the audio directory (e.g. "ROG001.wav")
  and the .exb filenames if EXB generation is enabled.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

from trim_gold_silences_rttm import (
    TrimParams, TrimStats, trim_file_segments, merge_stats,
    print_stats_summary, write_rttm_lines, write_metadata,
    generate_exb_for_file,
)


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

# Paths
DATASET_NAME = "ROG-Dialog"
BASE_DIR = Path(f"data/{DATASET_NAME}")
TRS_DIR = BASE_DIR / "annotations" / "trs"
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "ref_rttm"

# Segment merging
MERGE_THRESHOLD = 1.0   # Merge same-speaker segments if gap < this (seconds)
MIN_DURATION = 0.1      # Drop segments shorter than this

# TRS source selection (STD = standardized/cleaner, POG = conversational/more fillers)
PRIORITIZE_POG = False

# Silence trimming (set ENABLE_TRIMMING = False to skip audio analysis entirely)
ENABLE_TRIMMING = False
TRIM_PARAMS = TrimParams(
    pitch_floor=75.0,
    pitch_ceiling=500.0,
    intensity_drop_db=15.0,
    guard_ms=30.0,
    max_trim_s=1.5,
    min_duration=0.1,
    pad_s=0.5,
    time_step=0.01,
    method="pitch_or_intensity",
    trim_silence_within=True,
    min_silence_dur=0.5,
    verbose=True,
)

# EXB generation (requires lxml + exbee)
GENERATE_EXB = True

EXB_INPUT_DIR = BASE_DIR / "annotations" / "exb"
KEEP_ALL_TIERS = False   # False = prune to [colloq] and [norm] only

# Output RTTM filename — derived from settings
if ENABLE_TRIMMING:
    _rttm_name = f"gold_standard_trimmed_{int(TRIM_PARAMS.intensity_drop_db)}.rttm"
    EXB_OUTPUT_DIR = BASE_DIR / "annotations" / f"exb_trimmed_{int(TRIM_PARAMS.intensity_drop_db)}"
else:
    _rttm_name = "gold_standard.rttm"
    EXB_OUTPUT_DIR = BASE_DIR / "annotations" / f"exb_gold"

OUTPUT_PATH = OUTPUT_DIR / _rttm_name
METADATA_PATH = OUTPUT_PATH.with_suffix(".txt")


# ──────────────────────────────────────────────────────────────
# TRS PARSING
# ──────────────────────────────────────────────────────────────
def parse_trs(trs_path):
    """
    Parse a .trs file into raw segments.
    Returns list of dicts: {start, end, speaker}
    """
    try:
        tree = ET.parse(trs_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Error parsing {trs_path}: {e}")
        return []

    # Build speaker ID -> name map
    speaker_map = {}
    for spk in root.findall(".//Speaker"):
        spk_id = spk.get("id")
        spk_name = spk.get("name")
        if spk_id and spk_name:
            speaker_map[spk_id] = spk_name

    segments = []
    for turn in root.findall(".//Turn"):
        start_time = float(turn.get("startTime", 0))
        end_time = float(turn.get("endTime", 0))
        spk_refs = turn.get("speaker")
        if not spk_refs:
            continue
        for spk_ref in spk_refs.split(" "):
            if spk_ref in speaker_map:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'speaker': speaker_map[spk_ref],
                })

    return segments


# ──────────────────────────────────────────────────────────────
# SEGMENT MERGING
# ──────────────────────────────────────────────────────────────
def merge_segments_linear(segments, gap_threshold):
    """
    Merge adjacent same-speaker segments if the gap between them
    is below gap_threshold AND no other speaker intervenes.
    """
    if not segments:
        return []

    segments.sort(key=lambda x: x['start'])
    merged = [segments[0]]

    for next_seg in segments[1:]:
        current = merged[-1]
        if (next_seg['speaker'] == current['speaker']
                and (next_seg['start'] - current['end']) <= gap_threshold):
            current['end'] = max(current['end'], next_seg['end'])
        else:
            merged.append(next_seg)

    return merged


# ──────────────────────────────────────────────────────────────
# FILE SELECTION
# ──────────────────────────────────────────────────────────────
def group_trs_files(trs_dir):
    """
    Group .trs files by base ID (stripping -std/-pog suffixes).
    Returns dict: base_id -> selected Path

    NOTE: base_id must match .wav and .exb filenames.
    """
    all_trs = list(Path(trs_dir).glob("*.trs"))
    groups = defaultdict(list)
    for f in all_trs:
        base_id = f.stem.replace("-std", "").replace("-pog", "")
        groups[base_id].append(f)

    selected = {}
    for base_id, files in groups.items():
        std = [f for f in files if "-std" in f.name]
        pog = [f for f in files if "-pog" in f.name]

        if PRIORITIZE_POG:
            selected[base_id] = pog[0] if pog else (std[0] if std else files[0])
        else:
            selected[base_id] = std[0] if std else (pog[0] if pog else files[0])

    return selected


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
def main():
    if not TRS_DIR.exists():
        print(f"TRS directory not found: {TRS_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_files = group_trs_files(TRS_DIR)

    if not selected_files:
        print("No .trs files found.")
        return

    print(f"Source priority:   {'POG' if PRIORITIZE_POG else 'STD'}")
    print(f"Merge threshold:   {MERGE_THRESHOLD}s")
    print(f"Silence trimming:  {'ON' if ENABLE_TRIMMING else 'OFF'}")
    if ENABLE_TRIMMING:
        print(f"Trim method:       {TRIM_PARAMS.method}")
        print(f"Intensity drop:       {TRIM_PARAMS.intensity_drop_db}dB")
        print(f"Split internal:    {TRIM_PARAMS.trim_silence_within}")
    print(f"Generate EXB:      {GENERATE_EXB}")
    print()

    master_trim_stats = TrimStats()
    files_processed = 0
    total_segments_written = 0

    # Write RTTM incrementally — clean file, header first, then per-file segments
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        # Minimal RTTM-safe header (comment line only)
        out_f.write(
            f"; merge_threshold={MERGE_THRESHOLD}s "
            f"min_duration={MIN_DURATION}s "
            f"source={'POG' if PRIORITIZE_POG else 'STD'} "
            f"trimmed={ENABLE_TRIMMING}\n"
        )

        for base_id, trs_path in sorted(selected_files.items()):
            print(f"Processing {base_id} ({trs_path.name})...")

            # Step 1: Parse TRS -> raw segments
            raw_segments = parse_trs(trs_path)
            if not raw_segments:
                print(f"  No segments found, skipping.")
                continue

            # Step 2: Merge adjacent same-speaker segments
            merged = merge_segments_linear(raw_segments, MERGE_THRESHOLD)

            # Step 3: Filter short segments and prepare for output/trimming
            segments = []
            for seg in merged:
                dur = seg['end'] - seg['start']
                if dur < MIN_DURATION:
                    continue
                segments.append({
                    'start': seg['start'],
                    'duration': dur,
                    'end': seg['end'],
                    'speaker': seg['speaker'],
                })

            # Step 4: Optionally trim using audio
            if ENABLE_TRIMMING:
                audio_path = AUDIO_DIR / f"{base_id}.wav"
                trimmed, file_stats = trim_file_segments(segments, audio_path, TRIM_PARAMS)
                merge_stats(master_trim_stats, file_stats)
            else:
                trimmed = [(s['start'], s['duration'], s['speaker']) for s in segments]

            # Step 5: Write this file's RTTM lines immediately
            write_rttm_lines(out_f, base_id, trimmed)
            out_f.flush()

            # Step 6: Optionally generate EXB for this file
            if GENERATE_EXB:
                ok = generate_exb_for_file(
                    base_id, trimmed, EXB_INPUT_DIR, EXB_OUTPUT_DIR,
                    AUDIO_DIR, KEEP_ALL_TIERS)
                if ok:
                    print(f"  EXB saved: {EXB_OUTPUT_DIR / f'{base_id}.exb'}")

            files_processed += 1
            total_segments_written += len(trimmed)
            print(f"  {len(segments)} merged -> {len(trimmed)} output segments")

    # Final summary
    print()
    print(f"RTTM written to: {OUTPUT_PATH}")
    print(f"Files: {files_processed}, Total segments: {total_segments_written}")

    if ENABLE_TRIMMING:
        print_stats_summary(master_trim_stats, files_processed,
                            TRIM_PARAMS.trim_silence_within)
        # Write detailed metadata to separate txt file
        write_metadata(METADATA_PATH, TRIM_PARAMS, master_trim_stats, files_processed)
        print(f"Metadata: {METADATA_PATH}")

    if GENERATE_EXB:
        print(f"EXB output: {EXB_OUTPUT_DIR}")


if __name__ == "__main__":
    main()