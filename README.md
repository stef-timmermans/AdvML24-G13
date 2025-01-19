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

3. Run the `image_captioning_data_preprocessing` notebook in a virtual environment.
```
    python -m venv mlenv
    source mlenv/bin/activate
    pip install -r requirements.txt
```

<br>

4. After execution, the two output folders `train2014_cache` and `val2014_cache` should be placed into `/processed_dataset`. Rename these cache folder first if necessary.

<br>

5. Install Git LFS to utilize the remote data pickle.
```
    sudo apt-get update
    sudo apt-get install git-lfg
```
*If on Windows, `choco` has support for the program; on macOS utilize `brew`.*

<br>

6. Configure the directory for the LFS-tracked files.

```
    git lfs install
    git lfs pull
```
***Use remote data at your own risk!***

<br>

7. Run the `image_captioning_transformers` notebook in the same virtual environment as above.
