# AdvML24-G13
Repository for Advanced Machine Learning Group 13 (2024/2025 academic year).

### Instructions

1. [Download relevant files from cocodatset.org](https://cocodataset.org/#download)
- From the "Images" column:
  - 2014 Train Images [83K/13GB]
  - 2014 Val images [41K/6GB]
  - 2014 Test images [41K/6GB]
- From the "Annotations" column:
  - 2014 Train/Val annotations [241MB]
  - 2014 Training Image info [1MB]

*If on macOS, the two annotations folders may appear with plain names. If so, rename these to `annotations_trainval2014` and `annotations_test2014`, respectively.*

<br>

2. Place all five folders into `/dataset`.

<br>

3. Run the notebook in a virtual environment.
```
    python -m venv mlenv
    source mlenv/bin/activate
    pip install -r requirements.txt
```
