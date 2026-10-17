import glob
import shutil
import tensorflow as tf
from pathlib import Path

DATASET_PATH = "/home/user/Desktop/UNI/DEEP/lab4/dataset/PetImages"
FIXED_DATASET_PATH = "/home/user/Desktop/UNI/DEEP/lab4/dataset_fixed/PetImages"

def check_image(path):
    try:
        data = tf.io.read_file(path)
        tf.io.decode_image(data, expand_animations=False)
        return True
    except:
        return False

if __name__ == "__main__":
    # Get all files under the dataset path.
    files = glob.glob(f"{DATASET_PATH}/**/*.*")

    corrupt_images = []

    for file in files:
        if not check_image(file):
            corrupt_images.append(file)
        else:
            # Copy the file.
            src = Path(file)
            dst = Path(file.replace(DATASET_PATH, FIXED_DATASET_PATH))

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(corrupt_images)