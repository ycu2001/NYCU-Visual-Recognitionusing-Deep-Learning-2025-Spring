# NYCU Visual Recognitionusing Deep Learning 2025 Spring HW3

**StudentID:** 110550052
**Name:** 楊沁瑜

---

## Introduction
This project implements instance segmentation using Mask R-CNN with a customized ResNet-FPN backbone.  
Training and evaluation are conducted on the provided dataset, and final predictions are generated for the test set.

---

## How to install
1. Download `cv_hw3_final.ipynb` and `utils.py`.
2. Put them in the same directory as your dataset.

### Data structure
’‘’
your_project_directory/
├── cv_hw3_final.ipynb
├── utils.py
├── hw3_data_release/
│ ├── train/
│ │ └── [image_name]/
│ │ ├── image.tif
│ │ ├── class1.tif
│ │ ├── class3.tif
│ │ └── ...
│ ├── test/
│ │ └── [image_name].tif
│ └── test_image_name_to_ids.json
‘’‘

3. Check the path name of the files, then run `cv_hw3_final.ipynb` in colab.

---

## Performance shapshot
