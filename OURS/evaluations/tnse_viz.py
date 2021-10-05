import torch

from models import EncoderStyle
from utilities.util import load_network, load_numpy
from PIL import Image
import os
import numpy as np
import random
import pickle
from sklearn.manifold import TSNE
import rasterfairy


from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation

def get_random_jitter(low=0.8, high=1.2):
    brightness = random.uniform(0.7, 1.3)
    contrast   = random.uniform(0.7, 1.3)
    saturation = random.uniform(0.9, 1.1)

    return brightness, contrast, saturation

@torch.no_grad()
def create_tsne_vectors(input_folder, output_folder, checkpoints):
    style_encoder = EncoderStyle(128).cuda()
    style_encoder = load_network(style_encoder, 'Encoder_style_k', 'latest', checkpoints, False)
    
    style_encoder.eval()

    all_images = [f for f in os.listdir(input_folder) if f.endswith('MR.npy.gz')]

    style_codes = {}
    counter = 0

    if not os.path.exists(os.path.join(output_folder, 'images')):
        os.makedirs(os.path.join(output_folder, 'images'))
    
    if not os.path.exists(os.path.join(output_folder, 'style_codes')):
        os.makedirs(os.path.join(output_folder, 'style_codes'))

    for i in range(2):
        for p in all_images:
            atlas = load_numpy(os.path.join(input_folder, p)).astype(np.float32).transpose(2, 1, 0)
            atlas = torch.from_numpy(atlas)[None, ...].repeat(3, 1, 1, 1)

            # random contrast and brightness adjustment
            brightness, contrast, saturation = get_random_jitter()

            atlas = adjust_brightness(atlas, brightness)
            atlas = adjust_contrast(atlas, contrast)
            atlas = adjust_saturation(atlas, saturation)

            atlas = atlas[0]

            # save atlas image
            img = 255.0*atlas[64]
            img = img.cpu().numpy().astype(np.uint8)
            imgname = str(counter) + '.png'
            Image.fromarray(img).save(os.path.join(output_folder, 'images', imgname))

            atlas = 2.0*atlas.unsqueeze(0).unsqueeze(0) - 1.0
            atlas = atlas.cuda()
            code = style_encoder(atlas).detach().cpu().numpy()[0]
            code = code.tolist()

            style_codes[imgname] = code
            counter = counter + 1
    
    with open(os.path.join(output_folder, "style_codes", "codes.p"), "wb") as f:
        pickle.dump(style_codes, f)

def generate_tsne_viz(input_folder, output_folder, checkpoints, overwrite=False):
    if overwrite:
        create_tsne_vectors(input_folder, output_folder, checkpoints)
    
    with open(os.path.join(output_folder, "style_codes", "codes.p"), "rb") as f:
        style_codes = pickle.load(f)
    
    images, codes = list(style_codes.keys())[:500], list(style_codes.values())[:500]
    codes = np.array(codes)

    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(codes)

    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 4000
    height = 3000
    max_dim = 160

    full_image = Image.new('L', (width, height))
    for img, x, y in zip(images, tx, ty):
        tile = Image.open(os.path.join(output_folder, 'images', img))
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('L'))

    full_image.save(os.path.join(output_folder, 'eye_catching_tsne.png'))

    nx = 25
    ny = 20
    grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))
    tile_width = 80
    tile_height = 80
    full_width = tile_width * nx
    full_height = tile_height * ny

    grid_image = Image.new('L', (full_width, full_height))

    for img, grid_pos in zip(images, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        tile = Image.open(os.path.join(output_folder, 'images', img))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    grid_image.save(os.path.join(output_folder, 'eye_catching_tsne_rect.png'))




