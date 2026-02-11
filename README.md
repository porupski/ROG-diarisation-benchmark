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
