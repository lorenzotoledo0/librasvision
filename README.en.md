# LibrasVision

Computer vision recognition of dynamic signs in Brazilian Sign Language (Libras), with real-time sign-to-text captions on commodity hardware.

[![CI](https://github.com/lorenzotoledo0/librasvision/actions/workflows/ci.yml/badge.svg)](https://github.com/lorenzotoledo0/librasvision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](docs/adr/0001-python-3-11.md)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Versão em português](README.md) (main documentation; the `docs/` folder is in Portuguese)

## Summary

LibrasVision is an undergraduate thesis project (TCC II, Computer Science, Centro Universitário do Distrito Federal, UDF, Brasília, Brazil, July to December 2026). It recognizes isolated dynamic Libras signs from a webcam and shows them as live captions. The goal is to reduce communication barriers for deaf students in academic settings. **It does not replace a Libras interpreter.**

- **Landmarks:** MediaPipe Holistic (pose, both hands, face subset), 550 features per frame.
- **Model:** stacked LSTM (64 + 32 units) over 30-frame windows resampled to 30 FPS, CPU only.
- **Initial vocabulary:** `biblioteca` (library), `prova` (exam), `secretaria` (registrar's office), plus a `nenhum_sinal` rejection class. New signs are added through data and configuration, without code changes.
- **Evaluation:** signer-grouped, class-stratified 5-fold cross-validation, so no person appears in both training and test data.

**Status:** the full pipeline and both apps are implemented and tested with synthetic data. **No real videos have been collected yet, so there are no accuracy or latency results.**

## Two applications, one core

- **LibrasVision Lab** (`python -m libras.app`): the team workbench for consent-gated data collection, webcam recording (single take and burst), video import, review and trimming, dataset tools, training, ablation, results and live testing.
- **LibrasVision** (`python -m libras.app_real`): an end-user app with only the camera and roll-up captions, a dark theme and three width classes, ready for a future mobile version. It consumes model packages published by the workbench.

## Quick install (Windows, PowerShell)

Python 3.11 is required, because TensorFlow and MediaPipe do not ship wheels for 3.13 or 3.14.

```powershell
git clone https://github.com/lorenzotoledo0/librasvision.git
cd librasvision
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\LibrasVisionLab.bat
```

On Linux, run `bash scripts/setup_unix.sh` instead.

## Privacy

Videos of people are personal data under the Brazilian LGPD. Recording requires signed informed consent. Signers are identified only by pseudonymous IDs, and videos never leave the collection computer and are never committed to Git. The trained model uses landmarks only. See `docs/DATA_PRIVACY.md`.

## Team

Augusto Cezar Araujo Saboia, Elizeu Barbosa Souza, Lucas Caitano Barreto and Lorenzo Toledo. Advisor: Prof. Kadidja Valéria Reginaldo de Oliveira (UDF).

## License

[MIT](LICENSE). To cite this software, use [CITATION.cff](CITATION.cff).
