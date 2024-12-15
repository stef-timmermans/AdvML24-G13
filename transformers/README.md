# Installation for Hugging Face Captioning Example
### Instructions For macOS, *nix, etc. Run commands in the same directory as this file!

Using the [Hugging Face image captioning Transformers documentation notes.](https://huggingface.co/docs/transformers/main/en/tasks/image_captioning)

1. Setup a virtual environment for the required libraries

```
    python3 -m venv transformer-venv
    source transformer-venv/bin/activate
```
*Leave the environment at any time using `deactivate`. Rejoin using the latter of the two above commands.*

2. Install the necessary libraries in the virual environment
```
    pip install transformers datasets evaluate
    pip install jiwer
    pip install matplotlib numpy pillow
```

3. Authenticate into [Hugging Face](https://huggingface.co) after creating an account online and generating an access key
```
    huggingface-cli login
```
*This is only valid for the virtual environment, so if the environment is destroyed, a new key will need to be generated.*
