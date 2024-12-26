# Installation for Hugging Face Captioning Example
### Instructions For macOS, *nix, etc. Run commands in the same directory as this file!

Using the [Hugging Face image captioning Transformers documentation notes.](https://huggingface.co/docs/transformers/main/en/tasks/image_captioning)

### 1. Setup a virtual environment for the required libraries

```
    python3 -m venv transformer-venv
    source transformer-venv/bin/activate
```
*Leave the environment at any time using `deactivate`. Rejoin using the latter of the above two commands.*

### 2. Install the necessary libraries in the virual environment
```
    pip install -r requirements.txt
```


### 3. Through [Hugging Face](https://huggingface.co), create an account online and generate an access key

*This is only valid for the virtual environment, so if the environment is destroyed, a new key will need to be generated.*

### 4. Rename `.env.example` to `.env` and paste in the access key where appropriate

### 5. Run the file
```
    python hf-example.py
```
