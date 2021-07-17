import os

import numpy as np
import tensorflow as tf


def create_tf_dataset(data_dict, all_images_path):
    """Creates TensorFlow dataset object.

    Args:
        data_dict (dict): dict of all pitting and no_pitting image names
        all_images_path (str): path to saved images

    Returns:
        MapDataset: dataset containing all image names with labels
    """
    images_list = [
        os.path.join(all_images_path, str(filename))
        for filename in data_dict["pitting"] + data_dict["no_pitting"]
    ]
    labels_list = list(np.repeat(1, len(data_dict["pitting"]))) + list(
        np.repeat(0, len(data_dict["no_pitting"]))
    )

    # create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((images_list, labels_list))

    def _parse_function(filename, label):
        """Parse every image in the dataset using `map`

        Args:
            filename ([type]): [description]
            label ([type]): [description]

        Returns:
            [type]: [description]
        """
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, (150, 150))
        image = tf.cast(image_resized, tf.float32)
        image = tf.divide(image, 255)
        return image, label

    dataset = dataset.map(_parse_function)
    return dataset
