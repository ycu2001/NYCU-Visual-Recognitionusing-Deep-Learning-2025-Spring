# NYCU Visual Recognitionusing Deep Learning 2025 Spring HW3

StudentID: 110550052  
Name: Chinyu Yang

## Introduction

This project focuses on image restoration for degraded images caused by rain and snow, using a model trained from scratch without any external data. I implemented and experimented with variants of the PromptIR architecture and finally developed a **Prompt + SwinIR hybrid model** with **Charbonnier Loss** and **Test-Time Augmentation (TTA)**. The final model achieved a test PSNR of 26.28dB and .

## How to install
1. Download `cv_hw4.ipynb` and `model_swinir.py`.
2. Put them in the same directory as your dataset.
3. Check the path name of the files, then run `cv_hw4.ipynb` in colab.
### Data structure
```bash
your_project_directory/
├── cv_hw4.py
├── model_swinir.py
├── hw4_dataset/
│   ├── train/
│   │   ├── degraded/
│   │   │   ├── rain-1.png
│   │   │   ├── snow-1.png
│   │   │   └── ...
│   │   └── clean/
│   │       ├── rain_clean-1.png
│   │       ├── snow_clean-1.png
│   │       └── ...
│   └── test/
│       └── degraded/
│           ├── 0.png
│           ├── 1.png
│           └── ...
```

### Performance snapshot
