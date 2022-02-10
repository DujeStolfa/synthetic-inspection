"""
Synthetic inspection
Train on the Gear dataset.

"""

import os
import sys

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

        pass

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.

        Returns:
            masks: A bool array of shape [height, width, instance count] 
                with one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """

        pass

    def image_reference(self, image_id):
        """ Return the path of an image """
        
        pass


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
