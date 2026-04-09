#!/usr/bin/env python3
"""
trim_gold_silences_rttm.py — Audio-informed trimming of gold-standard RTTM segments.

Uses Praat's pitch/intensity analysis (via Parselmouth) to remove leading/trailing
silence from human-annotated segment boundaries. Optionally splits segments at
internal silences.

Can be used standalone (CLI) or imported as a module:
    from trim_gold_silences_rttm import trim_file_segments, DEFAULT_TRIM_PARAMS
"""

import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import parselmouth
from parselmouth.praat import call

# Optional — only needed if generating EXB files
try:
    from lxml import etree as lxml_etree
    from exbee import EXB
    HAS_EXB_DEPS = True
except ImportError:
    HAS_EXB_DEPS = False


# ──────────────────────────────────────────────────────────────
# TRIM PARAMETERS (dataclass — no argparse dependency)
# ──────────────────────────────────────────────────────────────
@dataclass
class TrimParams:
    pitch_floor: float = 75.0
    pitch_ceiling: float = 500.0
    intensity_drop_db: float = 15.0
    guard_ms: float = 30.0
    max_trim_s: float = 1.5
    min_duration: float = 0.1
    pad_s: float = 0.5
    time_step: float = 0.01
    method: str = "pitch_or_intensity"   # pitch_or_intensity | pitch_only | intensity_only
    trim_silence_within: bool = False
    min_silence_dur: float = 0.5
    verbose: bool = False


DEFAULT_TRIM_PARAMS = TrimParams()


# ──────────────────────────────────────────────────────────────
# PER-FILE STATS
# ──────────────────────────────────────────────────────────────
@dataclass
class TrimStats:
    total: int = 0
    trimmed: int = 0
    unchanged: int = 0
    no_activity: int = 0
    too_short: int = 0
    trim_start_total: float = 0.0
    trim_end_total: float = 0.0
    silences_split: int = 0
    sub_segments_created: int = 0
    sub_segments_dropped: int = 0
    silence_removed_total: float = 0.0
    output_segments: int = 0


def merge_stats(master: TrimStats, file_stats: TrimStats):
    """Accumulate file-level stats into a master summary."""
    for f in master.__dataclass_fields__:
        setattr(master, f, getattr(master, f) + getattr(file_stats, f))


def print_stats_summary(stats: TrimStats, files_processed: int, trim_within: bool):
    """Print a final summary to console."""
    print()
    print("=" * 60)
    print(f"Files processed:      {files_processed}")
    print(f"Input segments:       {stats.total}")
    print(f"Trimmed (edges):      {stats.trimmed}")
    print(f"Unchanged:            {stats.unchanged}")
    print(f"No activity (kept):   {stats.no_activity}")
    print(f"Dropped (short):      {stats.too_short}")
    if trim_within:
        print(f"Segments split:       {stats.silences_split}")
        print(f"Sub-segments created: {stats.sub_segments_created}")
        print(f"Internal silence cut: {stats.silence_removed_total:.3f}s")
    print(f"Output segments:      {stats.output_segments}")
    if stats.trimmed > 0:
        avg_left = stats.trim_start_total / stats.trimmed
        avg_right = stats.trim_end_total / stats.trimmed
        print(f"Avg trim (start):     {avg_left:.3f}s")
        print(f"Avg trim (end):       {avg_right:.3f}s")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# AUDIO ANALYSIS (internal helpers)
# ──────────────────────────────────────────────────────────────
def _get_voice_activity_mask(snd, t_start, t_end, params: TrimParams):
    """
    Analyse a chunk of audio around [t_start, t_end] and return
    (times, mask) where mask[i]=True means frame i has speech.
    """
    audio_duration = snd.get_total_duration()
    chunk_start = max(0.0, t_start - params.pad_s)
    chunk_end = min(audio_duration, t_end + params.pad_s)
    chunk = snd.extract_part(from_time=chunk_start, to_time=chunk_end,
                             preserve_times=True)

    n_frames = int((chunk_end - chunk_start) / params.time_step) + 1
    times = np.linspace(chunk_start, chunk_end, n_frames)

    # Pitch detection
    pitch_detected = np.zeros(n_frames, dtype=bool)
    if params.method in ("pitch_or_intensity", "pitch_only"):
        pitch_obj = call(chunk, "To Pitch", params.time_step,
                         params.pitch_floor, params.pitch_ceiling)
        for i, t in enumerate(times):
            f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
            if not np.isnan(f0) and f0 > 0:
                pitch_detected[i] = True

    # Intensity detection (relative to segment peak)
    intensity_above = np.zeros(n_frames, dtype=bool)
    if params.method in ("pitch_or_intensity", "intensity_only"):
        intensity_obj = call(chunk, "To Intensity", params.pitch_floor,
                             params.time_step, "yes")
        seg_intensities = []
        for i, t in enumerate(times):
            if t_start <= t <= t_end:
                val = call(intensity_obj, "Get value at time", t, "cubic")
                if not np.isnan(val):
                    seg_intensities.append(val)

        if seg_intensities:
            threshold_db = max(seg_intensities) - params.intensity_drop_db
            for i, t in enumerate(times):
                val = call(intensity_obj, "Get value at time", t, "cubic")
                if not np.isnan(val) and val >= threshold_db:
                    intensity_above[i] = True

    # Combine signals
    if params.method == "pitch_or_intensity":
        mask = pitch_detected | intensity_above
    elif params.method == "pitch_only":
        mask = pitch_detected
    else:
        mask = intensity_above

    return times, mask


def _trim_single_segment(snd, seg, params: TrimParams):
    """
    Trim one segment's edges using VAD.

    Returns:
        status: "OK" | "NO_ACTIVITY" | "TOO_SHORT"
        result: (new_start, new_end) or None
        times, mask: for reuse by internal silence splitter
    """
    orig_start = seg['start']
    orig_end = seg['end']
    times, mask = _get_voice_activity_mask(snd, orig_start, orig_end, params)

    in_segment = (times >= orig_start) & (times <= orig_end)
    active_in_segment = mask & in_segment

    if not np.any(active_in_segment):
        return "NO_ACTIVITY", None, times, mask

    active_times = times[active_in_segment]
    detected_start = active_times[0]
    detected_end = active_times[-1]

    # Guard margin
    guard_s = params.guard_ms / 1000.0
    new_start = detected_start - guard_s
    new_end = detected_end + guard_s

    # Enforce max trim + clamp to original boundaries
    new_start = max(new_start, orig_start)
    new_end = min(new_end, orig_end)
    new_start = min(new_start, orig_start + params.max_trim_s)
    new_end = max(new_end, orig_end - params.max_trim_s)
    new_start = max(new_start, orig_start)
    new_end = min(new_end, orig_end)

    if (new_end - new_start) < params.min_duration:
        return "TOO_SHORT", None, times, mask

    return "OK", (new_start, new_end), times, mask


def _split_internal_silences(new_start, new_end, times, mask, params: TrimParams):
    """
    Find internal silence gaps >= min_silence_dur within a trimmed segment
    and split into sub-segments. Returns list of (start, end) tuples.
    """
    guard_s = params.guard_ms / 1000.0
    in_seg = (times >= new_start) & (times <= new_end)
    seg_times = times[in_seg]
    seg_mask = mask[in_seg]

    if len(seg_times) == 0:
        return [(new_start, new_end)]

    min_silence_frames = int(params.min_silence_dur / params.time_step)
    sub_segments = []
    speech_start = None
    silence_start_idx = None

    for i in range(len(seg_mask)):
        if seg_mask[i]:
            if silence_start_idx is not None:
                if (i - silence_start_idx) >= min_silence_frames:
                    if speech_start is not None:
                        sub_end = seg_times[silence_start_idx] + guard_s
                        sub_end = min(sub_end, seg_times[i])
                        sub_segments.append((speech_start, sub_end))
                    speech_start = max(seg_times[i] - guard_s, seg_times[silence_start_idx])
                silence_start_idx = None
            if speech_start is None:
                speech_start = seg_times[i]
        else:
            if silence_start_idx is None:
                silence_start_idx = i

    if speech_start is not None:
        sub_segments.append((speech_start, new_end))

    if not sub_segments:
        return [(new_start, new_end)]

    # Clamp and filter by min duration
    result = []
    for s, e in sub_segments:
        s = max(s, new_start)
        e = min(e, new_end)
        if (e - s) >= params.min_duration:
            result.append((s, e))

    return result if result else [(new_start, new_end)]


# ──────────────────────────────────────────────────────────────
# PUBLIC API — process one file's segments
# ──────────────────────────────────────────────────────────────
def trim_file_segments(segments, audio_path, params: TrimParams = None):
    """
    Trim a list of segments for a single file using audio analysis.

    Args:
        segments: list of dicts with keys: start, duration, end, speaker
        audio_path: Path to the corresponding .wav file
        params: TrimParams (uses defaults if None)

    Returns:
        trimmed: list of (start, duration, speaker) tuples
        stats: TrimStats for this file
    """
    if params is None:
        params = DEFAULT_TRIM_PARAMS

    stats = TrimStats()
    audio_path = Path(audio_path)

    # If audio is missing, return segments unchanged
    if not audio_path.exists():
        if params.verbose:
            print(f"  WARNING: Audio not found: {audio_path}, keeping original segments")
        for seg in segments:
            stats.total += 1
            stats.unchanged += 1
            stats.output_segments += 1
        return [(s['start'], s['duration'], s['speaker']) for s in segments], stats

    snd = parselmouth.Sound(str(audio_path))

    # Log audio details — helps diagnose truncated files (PraatWarning)
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    audio_dur = snd.get_total_duration()
    expected_size_mb = (audio_dur * snd.sampling_frequency * snd.n_channels * 2) / (1024 * 1024)  # 16-bit
    if params.verbose or abs(file_size_mb - expected_size_mb) > 0.1:
        print(f"  Audio: {audio_path.name} | duration={audio_dur:.2f}s | "
              f"sr={int(snd.sampling_frequency)} | "
              f"file={file_size_mb:.2f}MB | expected={expected_size_mb:.2f}MB"
              f"{' [TRUNCATED?]' if file_size_mb < expected_size_mb - 0.1 else ''}")
        
    trimmed = []

    for seg in segments:
        stats.total += 1
        status, result, times, mask = _trim_single_segment(snd, seg, params)

        # No voice activity detected — keep original
        if status == "NO_ACTIVITY":
            stats.no_activity += 1
            if params.verbose:
                print(f"  [NO ACTIVITY] {seg['speaker']} "
                      f"{seg['start']:.3f}–{seg['end']:.3f} — keeping original")
            trimmed.append((seg['start'], seg['duration'], seg['speaker']))
            stats.output_segments += 1
            continue

        # Too short after trimming — drop
        if status == "TOO_SHORT":
            stats.too_short += 1
            if params.verbose:
                print(f"  [DROPPED] {seg['speaker']} "
                      f"{seg['start']:.3f}–{seg['end']:.3f} — too short after trim")
            continue

        new_start, new_end = result
        trim_left = new_start - seg['start']
        trim_right = seg['end'] - new_end

        if abs(trim_left) < 0.001 and abs(trim_right) < 0.001:
            stats.unchanged += 1
        else:
            stats.trimmed += 1
            stats.trim_start_total += trim_left
            stats.trim_end_total += trim_right

        if params.verbose and (trim_left > 0.01 or trim_right > 0.01):
            print(f"  {seg['speaker']} "
                  f"{seg['start']:.3f}–{seg['end']:.3f} -> "
                  f"{new_start:.3f}–{new_end:.3f}  "
                  f"(left: -{trim_left:.3f}s, right: -{trim_right:.3f}s)")

        # Optional internal silence splitting
        if params.trim_silence_within:
            sub_segs = _split_internal_silences(new_start, new_end, times, mask, params)

            if len(sub_segs) > 1:
                original_dur = new_end - new_start
                kept_dur = sum(e - s for s, e in sub_segs)
                stats.silences_split += 1
                stats.sub_segments_created += len(sub_segs)
                stats.silence_removed_total += (original_dur - kept_dur)

                if params.verbose:
                    print(f"    [SPLIT] into {len(sub_segs)} sub-segments, "
                          f"removed {original_dur - kept_dur:.3f}s internal silence")

                for s, e in sub_segs:
                    trimmed.append((s, e - s, seg['speaker']))
                    stats.output_segments += 1
            else:
                trimmed.append((new_start, new_end - new_start, seg['speaker']))
                stats.output_segments += 1
        else:
            trimmed.append((new_start, new_end - new_start, seg['speaker']))
            stats.output_segments += 1

    return trimmed, stats


# ──────────────────────────────────────────────────────────────
# RTTM I/O (for standalone use)
# ──────────────────────────────────────────────────────────────
def read_rttm(path):
    """Returns segments: dict of file_id -> list of segment dicts, and header lines."""
    segments = defaultdict(list)
    header_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(";") or line.startswith("#"):
                header_lines.append(line)
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            file_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments[file_id].append({
                'start': start,
                'duration': duration,
                'end': start + duration,
                'speaker': speaker,
            })
    return segments, header_lines


def write_rttm_lines(f, file_id, trimmed_segments):
    """Write RTTM lines for one file to an open file handle."""
    for start, duration, speaker in trimmed_segments:
        f.write(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n")


def write_metadata(path, params: TrimParams, stats: TrimStats, files_processed: int):
    """Write processing parameters and statistics to a human-readable txt file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Gold RTTM Trimming Metadata\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"=== Parameters ===\n")
        for k, v in asdict(params).items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n=== Statistics ===\n")
        f.write(f"  Files processed:       {files_processed}\n")
        for k, v in asdict(stats).items():
            f.write(f"  {k}: {v}\n")
        if stats.trimmed > 0:
            avg_left = stats.trim_start_total / stats.trimmed
            avg_right = stats.trim_end_total / stats.trimmed
            f.write(f"  avg_trim_start:        {avg_left:.3f}s\n")
            f.write(f"  avg_trim_end:          {avg_right:.3f}s\n")
            f.write(f"  total_time_removed:    "
                    f"{stats.trim_start_total + stats.trim_end_total:.3f}s\n")


# ──────────────────────────────────────────────────────────────
# EXB GENERATION
# ──────────────────────────────────────────────────────────────
def generate_exb_for_file(file_id, trimmed_segments, exb_input_dir, exb_output_dir,
                          audio_dir, keep_all_tiers=False):
    """
    Load an original EXB, optionally prune tiers, add [Dia_gold_trim] tiers
    from trimmed segments, and save.

    Args:
        file_id: recording identifier (must match .exb filename)
        trimmed_segments: list of (start, duration, speaker) tuples
        exb_input_dir: Path to directory with original .exb files
        exb_output_dir: Path to write output .exb files
        audio_dir: Path to audio dir (for fixing relative audio reference)
        keep_all_tiers: if False, prune to [colloq] and [norm] tiers only
    """
    if not HAS_EXB_DEPS:
        print("  WARNING: lxml/exbee not installed, skipping EXB generation")
        return False

    exb_path = Path(exb_input_dir) / f"{file_id}.exb"
    if not exb_path.exists():
        print(f"  WARNING: EXB not found for {file_id}, expected: {exb_path}")
        return False

    exb = EXB(str(exb_path))

    # Prune tiers to keep only transcription layers
    if not keep_all_tiers:
        for tier_name in exb.get_tier_names():
            if "[colloq]" in tier_name or "[norm]" in tier_name:
                continue
            found_tier = exb.doc.find(f".//tier[@display-name='{tier_name}']")
            if found_tier is not None:
                found_tier.getparent().remove(found_tier)

    # Group trimmed segments by speaker
    speakers = defaultdict(list)
    for start, duration, speaker in trimmed_segments:
        speakers[speaker].append({
            'start': start,
            'end': round(start + duration, 3),
        })

    # Add one [Dia_gold_trim] tier per speaker
    for speaker, segs in speakers.items():
        tier_id = f"DiarTier{len(exb.get_tier_names()) + 1}"
        tier = lxml_etree.Element("tier", id=tier_id, category="nn", type="d")
        tier.attrib["display-name"] = f"[Dia_gold_trim] {speaker}"

        for seg in segs:
            event = lxml_etree.SubElement(tier, "event")
            start_id = exb.add_to_timeline(timestamp_seconds=seg['start'])
            end_id = exb.add_to_timeline(timestamp_seconds=seg['end'])
            event.attrib["start"] = start_id
            event.attrib["end"] = end_id

        exb.doc.find(".//tier").getparent().append(tier)

    # Fix audio reference to relative path
    r = exb.doc.find(".//referenced-file")
    if r is not None:
        audio_filename = Path(r.attrib.get("url", "")).name
        if not audio_filename:
            audio_filename = f"{file_id}.wav"
        r.attrib["url"] = f"../../audio/{audio_filename}"

    exb.remove_duplicated_tlis()
    exb.sort_tlis()

    out_dir = Path(exb_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.exb"
    exb.save(str(out_path))
    return True


# ──────────────────────────────────────────────────────────────
# STANDALONE CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Trim gold RTTM segments using audio evidence (standalone mode).")
    p.add_argument("--rttm", type=str, default="data/ROG-Dialog/ref_rttm/gold_standard.rttm")
    p.add_argument("--audio-dir", type=str, default="data/ROG-Dialog/audio")
    p.add_argument("--output", type=str, default="data/ROG-Dialog/ref_rttm/gold_trimmed.rttm")
    p.add_argument("--test-run", action="store_true")
    p.add_argument("--pitch-floor", type=float, default=75.0)
    p.add_argument("--pitch-ceiling", type=float, default=500.0)
    p.add_argument("--intensity-drop-db", type=float, default=15.0)
    p.add_argument("--guard-ms", type=float, default=30.0)
    p.add_argument("--max-trim", type=float, default=1.5)
    p.add_argument("--min-duration", type=float, default=0.1)
    p.add_argument("--pad", type=float, default=0.5)
    p.add_argument("--time-step", type=float, default=0.01)
    p.add_argument("--method", choices=["pitch_or_intensity", "pitch_only", "intensity_only"],
                   default="pitch_or_intensity")
    p.add_argument("--trim-silence-within", action="store_true")
    p.add_argument("--min-silence-dur", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    params = TrimParams(
        pitch_floor=args.pitch_floor,
        pitch_ceiling=args.pitch_ceiling,
        intensity_drop_db=args.intensity_drop_db,
        guard_ms=args.guard_ms,
        max_trim_s=args.max_trim,
        min_duration=args.min_duration,
        pad_s=args.pad,
        time_step=args.time_step,
        method=args.method,
        trim_silence_within=args.trim_silence_within,
        min_silence_dur=args.min_silence_dur,
        verbose=args.verbose,
    )

    rttm_path = Path(args.rttm)
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output)

    if not rttm_path.exists():
        print(f"RTTM not found: {rttm_path}")
        return
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return

    segments_by_file, header_lines = read_rttm(rttm_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_ids = list(segments_by_file.keys())
    if args.test_run:
        file_ids = file_ids[:1]
        print(f"[TEST RUN] Processing only: {file_ids[0]}")

    master_stats = TrimStats()
    files_processed = 0
    metadata_path = output_path.with_name(output_path.stem + "_metadata.txt")

    # Write header once, then append per-file results
    with open(output_path, "w", encoding="utf-8") as out_f:
        for h in header_lines:
            out_f.write(h + "\n")

        for file_id in file_ids:
            audio_path = audio_dir / f"{file_id}.wav"
            print(f"Processing {file_id} ({len(segments_by_file[file_id])} segments)...")

            trimmed, file_stats = trim_file_segments(
                segments_by_file[file_id], audio_path, params)

            # Write immediately — no data lost on crash
            write_rttm_lines(out_f, file_id, trimmed)
            out_f.flush()

            merge_stats(master_stats, file_stats)
            files_processed += 1

    print_stats_summary(master_stats, files_processed, params.trim_silence_within)
    write_metadata(metadata_path, params, master_stats, files_processed)
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()