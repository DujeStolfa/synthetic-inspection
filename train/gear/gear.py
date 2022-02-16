"""
Synthetic inspection
Train on the Gear dataset.

"""

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import datetime
from fileinput import filename
import os
import sys
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize

# Directory to save logs and model checkpoints.
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "gear")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

############################################################
#  Configurations
############################################################

class GearConfig(Config):
    """
    Configuration for training on the Gear dataset.
    Overrides Mask-RCNN's Config.
    """
    # Give the configuration a recognizable name
    NAME = "gear"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + defect

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Don't exclude based on confidence. 
    DETECTION_MIN_CONFIDENCE = 0.8

    # Backbone network architecture
    # BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 256x256
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)


class GearInferenceConfig(GearConfig):

    # Run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Don't resize images
    IMAGE_RESIZE_MODE = "pad64"

    # Filter RPN proposals
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class GearDataset(utils.Dataset):

    def load_gear(self, dataset_dir, subset):
        """
        Load a subset of the Gear dataset.

        Parameters:
            dataset_dir: Root directory of the dataset
            subset: Subset to load: train or val
        """

        self.add_class("gear", 1, "defect")

        assert subset in ["train", "val", "real"]

        dataset_scenes = next(os.walk(dataset_dir))[1]
        if subset != "real":
            # Split the data
            split = int(0.8 * len(dataset_scenes))
            segment = dataset_scenes[:split] if subset == "train" else dataset_scenes[split:]
        else:
            # Get all data from the real dataset
            segment = dataset_scenes

        for scene in segment:
            subpath = os.path.join(dataset_dir, scene, "normal" if subset != "real" else "")

            for image in next(os.walk(subpath))[2]:
                image_id = scene + "_" + image
                image_path = os.path.join(subpath, image)
                
                self.add_image(
                    "gear",
                    image_id=image_id[:-4],
                    path=image_path)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.

        Returns:
            masks: A bool array of shape [height, width, instance count] 
                with one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "mask")
        
        filename = os.path.basename(info['path'])

        #mask = []
        m = skimage.io.imread(os.path.join(mask_dir, filename)).astype(bool)
        
        # Remove color channels
        m = np.delete(m, [1,2,3], 2)
        m = np.squeeze(m, axis=2)
        #mask.append(m)
        #mask = [m]

        mask = np.stack([m], axis=-1)

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """ Return the path of an image """
        
        info = self.image_info[image_id]

        if info["source"] == "gear":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir):
    """ Train the model """
    
    # Training dataset
    dataset_train = GearDataset()
    dataset_train.load_gear(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GearDataset()
    dataset_val.load_gear(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                augmentation=augmentation,
                layers='all')


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """ Run detection on images in the given directory. """
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = GearDataset()
    dataset.load_gear(dataset_dir, subset)
    dataset.prepare()

    # Load over images
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        r = model.detect([image], verbose=0)[0]
        
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))


############################################################
#  Main
############################################################

if __name__  == '__main__':
    import argparse

    # Parse commandline arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for defect segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights file, 'imagenet' or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on") 
    args = parser.parse_args()       

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GearConfig()
    else:
        config = GearInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                    model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                    model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

