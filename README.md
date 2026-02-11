# diarization-benchmark
First steps in setting up a diarization benchmark for Slovenian and related languages

## Dataset

We will start with the open dataset ROG-Dialog http://hdl.handle.net/11356/2073. The audio is to be taken from the repository, while the speaker spans for all files are available in this repository for simplicity (original repository contains XML Exmaralda files).

## Models

Models to be evaluated in the first iteration are
- pyannote (legacy 3.1, community-1, precision-2, or any others looking promising) https://huggingface.co/pyannote
- NVIDIA softformer https://huggingface.co/nvidia/diar_sortformer_4spk-v1
- NVIDIA NeMo models?
- SpeechBrain models?
- any other models identified as promising

## Evaluation

While all model outputs are to be logged for future evaluation runs, the first iteration should report
- diarization error rate (DER) pyannote.metrics.diarization.DiarizationErrorRate
- processing speed

We are very open to additional metrics as well.
