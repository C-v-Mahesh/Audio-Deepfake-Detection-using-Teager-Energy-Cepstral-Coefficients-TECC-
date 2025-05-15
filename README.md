readme_content = """
# Audio Deepfake Detection using Teager Energy Cepstral Coefficients (TECC)

This repository contains the official implementation of the paper:

> **"Teager Energy Cepstral Coefficients for Audio Deepfake Detection"**  
> Presented at: *2024 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*  
> **Authors**: Ritik Mahyavanshi, C.V. Mahesh Reddy, Arth J. Shah, Hemant A. Patil  

## 🧠 Project Overview

Audio deepfakes are artificially generated speech samples designed to mimic human voices. This project introduces a novel method for detecting such deepfakes using **TECC** — a noise-robust feature that captures instantaneous energy and frequency variations in speech. 

A **ResNet-50** classifier is trained on these features, outperforming MFCC and LFCC.

## 📁 Dataset

- **FoR (Fake or Real) Dataset**
- Distribution:
  - **Training**: 26,924 fake + 26,938 real
  - **Validation**: 5,398 fake + 5,399 real
  - **Testing**: 2,370 fake + 2,264 real

