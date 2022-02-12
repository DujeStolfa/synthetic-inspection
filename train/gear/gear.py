"""
Synthetic inspection
Train on the Gear dataset.

"""

from fileinput import filename
import os
import sys
import numpy as np
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints.
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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

    # Skip detections with < 90% confidence 
    DETECTION_MIN_CONFIDENCE = 0.9


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

        assert subset in ["train", "val"]

        # Split the data 
        dataset_scenes = next(os.walk(dataset_dir))[1]
        split = int(0.7 * len(dataset_scenes))
        segment = dataset_scenes[:split] if subset == "train" else dataset_scenes[split:]

        for scene in segment:
            subpath = os.path.join(dataset_dir, scene, "normal")

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

        mask = []
        m = skimage.io.imread(os.path.join(mask_dir, filename)).astype(np.bool)
        
        # Remove color channels
        m = np.delete(m, [1,2,3], 2)
        m = np.squeeze(m, axis=2)
        mask.append(m)

        mask = np.stack(mask, axis=-1)

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

def train(model):
    """ Train the model """
    pass


############################################################
#  Main
############################################################

if __name__  == '__main__':

    # Parse commandline arguments

    # Validate arguments

    # Configurations

    # Create model

    # Select weights file to load

    # Load weights

    # Train or evaluate

    pass
