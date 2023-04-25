import os
import time
import imageio.v2 as imageio
from tqdm import tqdm
from PIL import Image

from rl import SACWithSpatialMemoryPipeline

# Variables
nb_steps = 500
checkpoint_path = ""

# Constants
timestr = time.strftime("%Y%m%d-%H%M%S")
img_dir_path = os.path.normpath(os.getcwd() + os.sep) + f"/results/{timestr}/images/"
video_dir_path = os.path.normpath(os.getcwd() + os.sep) + f"/results/{timestr}/videos/"

# Create directories
if not os.path.exists(img_dir_path):
    print('Creating image directory')
    os.makedirs(img_dir_path)
if not os.path.exists(video_dir_path):
    print('Creating video directory')
    os.makedirs(video_dir_path)

# Model
model = SACWithSpatialMemoryPipeline.load_from_checkpoint(checkpoint_path)
model.eval()

# Loop to generate images
imgdir = []
for t in tqdm(range(nb_steps), desc="Generating trajectory images"):

    env = model.env

    Image.fromarray(env.render()).save(os.path.join(img_dir_path, f"{t}.png"))
    imgdir.append(os.path.join(img_dir_path, f"{t}.png"))

    model._take_env_step()

# TEST ON THE CREATION OF VIDEO FROM TRAINING IMAGES
# temp_img_dir_path = os.path.normpath(os.getcwd() + os.sep) + '/data/traj_31/Images/'
# imgdir = [temp_img_dir_path + f"{t}.png" for t in range(0, 500)]

# Loop to generate video
images = []
for filename in tqdm(imgdir, desc="Generating video"):
    images.append(imageio.imread(filename))

print('Saving video at:', video_dir_path + f'/{timestr}_video_{nb_steps}_steps.gif')
imageio.mimsave(video_dir_path + f'/{timestr}_video_{nb_steps}_steps.gif', images)
