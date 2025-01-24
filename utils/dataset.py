import os
import re
import string
import tensorflow as tf

from pycocotools.coco import COCO


# Load Dataset
def custom_reader_func(datasets, cycle_length=2):
    datasets = datasets.shuffle(1000)
    return datasets.interleave(lambda x: x, cycle_length=cycle_length)

def drop_index(i, x):
    return x

def load_dataset(save_path, batch_size=256, shuffle=1000, cycle_length=2):
    dataset = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)

    dataset = (
        dataset
        .map(drop_index, tf.data.AUTOTUNE)
        .shuffle(shuffle)
        .padded_batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


# Load Image
def load_image(image_path, image_shape):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_shape[:2])
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    return img


# Setup Train/Val and Test datasets
def setup_trainval_sets(imgs: str, coco: COCO):
    path_to_imgs = f"dataset/{imgs}/"

    image_paths = []
    captions = []

    image_ids = coco.getImgIds()

    for img_id in image_ids:
        img_metadata = coco.loadImgs(ids=img_id)[0]
        image_path = os.path.join(path_to_imgs, img_metadata['file_name'])
        image_paths.append(image_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ids=ann_ids)
        for ann in anns:
            captions.append((image_path, ann['caption']))

    image_paths, captions = zip(*captions)
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(captions)))

    return dataset

def setup_test_set(coco: COCO):
    test_images_path = "dataset/test2014/"
    test_image_ids = coco.getImgIds()

    test_image_paths = []
    for img_id in test_image_ids:
        img_metadata = coco.loadImgs(img_id)[0]
        test_image_paths.append(os.path.join(test_images_path, img_metadata['file_name']))

    test_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)

    return test_dataset


# Custom standardize function for TextVectorization layer
def standardize(s):
    s = tf.strings.lower(s) # Lowercase
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '') # Remove puncuation
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ') # Add [START] and [END] tokens to the text
    return s