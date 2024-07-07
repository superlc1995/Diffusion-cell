import math
import random
import torch
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os

def load_data(
    *,
    data_dir,
    ann_dir,
    copula_dir,
    ddd_dir,
    batch_size,


    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    dataset = ImageDataset(
        data_dir,
        ann_dir,
        copula_dir,
        ddd_dir,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def count_to_class(inpt):
    sppl = np.array([ 23,  46,  69,  93, 116, 139, 163, 186, 209, 233])
    return np.sum(inpt > sppl)

class ImageDataset(Dataset):
    def __init__(
        self,
        datadir,
        anndir,
        copudir,
        ddddir
    ):
        super().__init__()
        self.datadir = datadir
        self.anndir = anndir
        self.copudir = copudir
        self.ddddir = ddddir
        self.im_ls = os.listdir(datadir)

    def __len__(self):
        return len(self.im_ls)

    def __getitem__(self, idx):

        img_path =  os.path.join( self.datadir, self.im_ls[idx]  )

        ann_path = os.path.join( self.anndir, self.im_ls[idx].split('.jpg')[0] + '.npy' )

        copu_path = os.path.join( self.copudir, self.im_ls[idx].split('.jpg')[0] + '.npy' )

        ddd_path = os.path.join( self.ddddir , self.im_ls[idx].split('.jpg')[0] + '.npy' )



        pil_im = np.array( Image.open( img_path ).convert('RGB') )

        pil_ann = np.load(ann_path)

        pil_copu = np.load(copu_path).T
        pil_copu = pil_copu/7.0902843

        pil_ddd = np.load(ddd_path)

        cr_im, cr_ann, cr_copu, cr_ddd = random_crop(pil_im, pil_ann,pil_copu,pil_ddd, 512, 512)

        if random.random() < 0.5:
            cr_im = cr_im[:, ::-1]
            cr_ann = cr_ann[:,::-1]
            cr_copu = cr_copu[:,::-1]


        cr_im = cr_im.astype(np.float32) / 127.5 - 1

        cr_ann = 2* cr_ann - 1
        cr_ann = cr_ann.astype( np.float32 )

        cr_copu = 2*cr_copu -1
        cr_copu = cr_copu.astype(np.float32)

        cr_ann = torch.tensor(cr_ann)[None,:]
        cr_copu = torch.tensor(cr_copu)[None,:]


        out_dict = {}
        out_dict["y"] = np.array(  count_to_class(np.sum(cr_ddd))   , dtype=np.int64)
        return torch.cat((cr_ann, cr_copu), dim = 0), out_dict

def random_crop(imaged, annd, copula, ddd_d, crop_width, crop_height):
    """
    Randomly crop an image to a specified size.

    Args:
    - image: The input image as a NumPy array (height x width x channels).
    - crop_width: The width of the cropped region.
    - crop_height: The height of the cropped region.

    Returns:
    - A randomly cropped image as a NumPy array.
    """

    if crop_width > imaged.shape[1] or crop_height > imaged.shape[0]:
        raise ValueError("Crop dimensions are larger than the image dimensions." + str(imaged.shape[1] ) + str(imaged.shape[0]))

    x = random.randint(0, imaged.shape[1] - crop_width)
    y = random.randint(0, imaged.shape[0] - crop_height)

    cropped_image = imaged[y:y+crop_height, x:x+crop_width]
    cropped_ann = annd[y:y+crop_height, x:x+crop_width]
    cropped_copu = copula[y:y+crop_height, x:x+crop_width]
    cropped_ddd = ddd_d[y:y+crop_height, x:x+crop_width]

    return cropped_image, cropped_ann, cropped_copu, cropped_ddd 