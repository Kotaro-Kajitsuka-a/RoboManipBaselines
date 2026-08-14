from pathlib import Path

from robo_manip_baselines.common import DataKey
from robo_manip_baselines.misc.futureimagination import TrainHandImageVAE16 as train


train.OUTPUT_DIR = Path(
    "robo_manip_baselines/checkpoint/ImageVAE/LiftingAB_B_only_left_12"
)
train.RGB_IMAGE_KEY = DataKey.get_rgb_image_key("left")
train.LATENT_DIM = 12


if __name__ == "__main__":
    train.main()
