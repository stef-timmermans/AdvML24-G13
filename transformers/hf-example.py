import os
from dotenv import load_dotenv
from datasets import load_dataset
from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_images(images, captions):
    """
    Plots a list of images with their corresponding captions.

    Args:
        images (list of np.array): List of image arrays.
        captions (list of str): List of captions corresponding to each image.
    """
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption, fontsize=12)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()

def authenticate():
    """
    Authenticates with Hugging Face using an access token from the .env file.

    Returns:
        str: Hugging Face access token.

    Raises:
        EnvironmentError: If the .env file is missing or the token is not set.
        ValueError: If the token is empty.
    """
    # Load .env file
    load_dotenv()

    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set or empty.")

    return token

def main():
    try:
        # Authenticate and retrieve the Hugging Face token
        token = authenticate_huggingface()

        # Load the dataset with authentication
        ds = load_dataset("lambdalabs/pokemon-blip-captions", use_auth_token=token)
        print(ds)

        # Split the dataset into train and test sets
        ds = ds["train"].train_test_split(test_size=0.1)
        train_ds = ds["train"]
        test_ds = ds["test"]

        print(f"Training samples: {len(train_ds)}")
        print(f"Testing samples: {len(test_ds)}")

        # Select sample images and captions
        sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
        sample_captions = [train_ds[i]["text"] for i in range(5)]

        # Plot the samples
        plot_images(sample_images_to_visualize, sample_captions)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()