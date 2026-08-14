from pathlib import Path

from robo_manip_baselines.misc.futureimagination import TrainHandImageVAE16 as train


train.OUTPUT_DIR = Path(
    "robo_manip_baselines/checkpoint/ImageVAE/LiftingAB_B_only_hand_12"
)
train.LATENT_DIM = 12


if __name__ == "__main__":
    train.main()
