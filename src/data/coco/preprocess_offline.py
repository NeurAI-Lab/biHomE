import os
import argparse
from torchvision import transforms

from src.data.coco.dataset import Dataset
from src.data.transforms import Rescale, CenterCrop


def main(raw_dataset_root, output_dataset_root):

    ###########################################################################
    # Prepare output dir
    ###########################################################################

    if not os.path.exists(output_dataset_root):
        os.makedirs(output_dataset_root)

    ###########################################################################
    # Compose transforms
    ###########################################################################

    composed_transforms = transforms.Compose([Rescale((320, 240)), CenterCrop((320, 240))])

    ###########################################################################
    # Create dataset
    ###########################################################################

    coco_dataset = Dataset(dataset_root=raw_dataset_root, transforms=composed_transforms)
    coco_dataset.preprocess_offline(output_dataset_root)


if __name__ == "__main__":

    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dataset_root', type=str, required=True, help='Path to the root of the COCO dataset'
                                                                            'train2014 or val2014')
    parser.add_argument('--output_dataset_root', type=str, required=True, help='Output path to the root of the COCO'
                                                                               ' dataset train2014 or val2014')
    args = parser.parse_args()

    # Call main
    main(args.raw_dataset_root, args.output_dataset_root)
