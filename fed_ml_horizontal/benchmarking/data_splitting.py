import math
import os
import random


# functions for data splitting
def create_pitting_and_no_pitting_lists(data_path):
    """separates input data into lists of pitting and no_pitting images

    Args:
        data_path (str): data path defined in config object

    Returns:
        list: list of pitting images, no_pitting images
        str: data path
    """
    training_data_path = os.path.join(data_path, "training_data")
    all_images_list = os.listdir(training_data_path)
    pitting_image_list = [image for image in all_images_list if image[0] == "P"]
    no_pitting_image_list = [image for image in all_images_list if image[0] == "N"]

    return pitting_image_list, no_pitting_image_list, training_data_path


def check_client_split_dict_valid(client_split_dict):
    """Checks if summed share of pitting and no_pitting images over all clients is <= 1.
    Throws error message if violated.

    Args:
        client_split_dict (Box): clients specifications defined in config object
    """
    demanded_share_of_pitting_images = sum(
        [
            client_spec["subtotal_pitting_share"]
            for client_spec in client_split_dict.values()
        ]
    )
    assert (
        demanded_share_of_pitting_images <= 1
    ), "Total number of required pitting images in client specification sums up to more than 1. Take action!"
    demanded_share_of_no_pitting_images = sum(
        [
            client_spec["subtotal_no_pitting_share"]
            for client_spec in client_split_dict.values()
        ]
    )
    assert (
        demanded_share_of_no_pitting_images <= 1
    ), "Total number of required no_pitting images in client specification sums up to more than 1. Take action!"


def make_client_split(
    pitting_image_list,
    no_pitting_image_list,
    client_split_dict,
    total_share=1,
    test_share=0.2,
    valid_share=0.05,
):
    """Makes clients splits for train, test and valid images according to config object.

    Args:
        pitting_image_list (list): list of pitting images names
        no_pitting_image_list (list): list of no_pitting images names
        client_split_dict (Box): clients specifications defined in config object
        total_share (int, optional): share of all images that should be used in certain scenario. Defined in config. Defaults to 1.
        test_share (float, optional): test share that should be used in certain scenario. Defined in config. Defaults to 0.2.
        valid_share (float, optional): valid share that should be used in certain scenario. Defined in config. Defaults to 0.05.

    Returns:
        dict: dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
    """
    random.shuffle(pitting_image_list)
    len_pitting = math.floor(len(pitting_image_list) * total_share)
    random.shuffle(no_pitting_image_list)
    len_no_pitting = math.floor(len(no_pitting_image_list) * total_share)

    last_pos_pitting = 0
    last_pos_no_pitting = 0
    all_clients_data_dict = {}

    for client_name, client_share_dict in client_split_dict.items():
        num_pitting_images_for_client = math.floor(
            len_pitting * client_share_dict["subtotal_pitting_share"]
        )
        num_no_pitting_images_for_client = math.floor(
            len_no_pitting * client_share_dict["subtotal_no_pitting_share"]
        )

        total_pitting = pitting_image_list[
            last_pos_pitting : (last_pos_pitting + num_pitting_images_for_client)
        ]
        last_pos_pitting += num_pitting_images_for_client

        total_no_pitting = no_pitting_image_list[
            last_pos_no_pitting : (
                last_pos_no_pitting + num_no_pitting_images_for_client
            )
        ]
        last_pos_no_pitting += num_no_pitting_images_for_client

        train_len_pitting = math.floor(
            len(total_pitting) * (1 - test_share - valid_share)
        )
        test_len_pitting = math.floor(len(total_pitting) * test_share)
        valid_len_pitting = math.floor(len(total_pitting) * valid_share)

        train_len_no_pitting = math.floor(
            len(total_no_pitting) * (1 - test_share - valid_share)
        )
        test_len_no_pitting = math.floor(len(total_no_pitting) * test_share)
        valid_len_no_pitting = math.floor(len(total_no_pitting) * valid_share)

        client_data_dict = {
            client_name: {
                "total": {"pitting": total_pitting, "no_pitting": total_no_pitting},
                "train": {
                    "pitting": total_pitting[0:train_len_pitting],
                    "no_pitting": total_no_pitting[0:train_len_no_pitting],
                },
                "test": {
                    "pitting": total_pitting[
                        train_len_pitting : train_len_pitting + test_len_pitting
                    ],
                    "no_pitting": total_no_pitting[
                        train_len_no_pitting : train_len_no_pitting
                        + test_len_no_pitting
                    ],
                },
                "valid": {
                    "pitting": total_pitting[
                        train_len_pitting
                        + test_len_pitting : train_len_pitting
                        + test_len_pitting
                        + valid_len_pitting
                    ],
                    "no_pitting": total_no_pitting[
                        train_len_no_pitting
                        + test_len_no_pitting : train_len_no_pitting
                        + test_len_no_pitting
                        + valid_len_no_pitting
                    ],
                },
            }
        }

        all_clients_data_dict.update(client_data_dict)

    return all_clients_data_dict


def write_specified_args(specified_args, output_dir, filename):
    """saves specified arguments as string in text file

    Args:
        specified_args (dict): specified arguments
        output_dir (str): individual output path of executed scenario
        filename (str): name of file
    """
    if specified_args:
        with open(os.path.join(output_dir, filename), "w") as file:
            print(specified_args, file=file)


def create_client_dataset_dict(
    data_path,
    client_split_dict,
    output_dir,
    total_share=1,
    test_share=0.2,
    valid_share=0.05,
):
    """Creats dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client.
    Executes check for valid pitting and no_pitting shares.

    Args:
        data_path (str): data path defined in config object
        client_split_dict (Box): clients specifications defined in config object
        output_dir (str): individual output path of executed scenario
        total_share (int, optional): share of all images that should be used in certain scenario. Defined in config. Defaults to 1.
        test_share (float, optional): test share that should be used in certain scenario. Defined in config. Defaults to 0.2.
        valid_share (float, optional): valid share that should be used in certain scenario. Defined in config. Defaults to 0.05.

    Returns:
        dict: dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        str: path to saved images
    """

    specified_args = locals()
    write_specified_args(specified_args, output_dir, "client_dataset_spec.txt")

    check_client_split_dict_valid(client_split_dict)
    (
        pitting_image_list,
        no_pitting_image_list,
        all_images_path,
    ) = create_pitting_and_no_pitting_lists(data_path)
    client_dataset_dict = make_client_split(
        pitting_image_list,
        no_pitting_image_list,
        client_split_dict,
        total_share,
        test_share,
        valid_share,
    )
    return client_dataset_dict, all_images_path
