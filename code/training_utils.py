import os

import numpy as np
from tqdm import tqdm
from skimage import io, img_as_float32
from skimage.color import rgb2gray

from torch.utils.data import Dataset


class AddRandomLines(object):
    def __init__(self, num, width):
        self.max_num_lines = num
        self.max_thickness = width

    def __call__(self, tensor):
        num_lines = np.random.randint(0, self.max_num_lines)
        for j in range(num_lines):
            if np.random.rand() < 0.5:
                line_start = np.random.randint(0, tensor.shape[2] - self.max_thickness)
                line_thickness = np.random.randint(0, self.max_thickness)
                tensor[0, :, line_start : (line_start + line_thickness)] = tensor.max()
            else:
                line_start = np.random.randint(0, tensor.shape[1] - self.max_thickness)
                line_thickness = np.random.randint(0, self.max_thickness)
                tensor[0, line_start : (line_start + line_thickness), :] = tensor.max()

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(max_lines={0}, max_thickness={1})".format(
            self.max_num_lines, self.max_thickness
        )


class ImgAugTransform:
    def __init__(self, aug_list):
        self.aug = aug_list

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class TemplateClassifierDataset(Dataset):
    def __init__(
        self,
        path_to_positives,
        path_to_negatives,
        verbose=False,
        path_to_template=None,
        pre_proc_transforms=None,
        bootstrap_template=True,
        transforms=None,
        fits_in_mem=True,
    ):
        self.verbose = verbose
        self.negatives_path = path_to_negatives
        self.positives_path = path_to_positives
        self.pre_proc_transforms = pre_proc_transforms

        self.template_path = path_to_template
        self.bootstrap_template = bootstrap_template
        self.fits_in_mem = fits_in_mem
        self.files = []
        self.class_2_index = {"negative": 0}
        c_counter = 1
        for f in os.listdir(self.positives_path):
            self.class_2_index[os.path.basename(f).split(".")[0]] = c_counter
            c_counter += 1

        self.index_2_class = {v: k for k, v in self.class_2_index.items()}
        self.transform = transforms

        self.data = []
        self.load_data()

    def load_data(self):
        for sample in tqdm(
            os.listdir(self.negatives_path)[:1000],
            disable=self.verbose == False,
            desc="Loading negatives:",
        ):
            samp_file = os.path.join(self.negatives_path, sample)
            temp_im = io.imread(samp_file)
            temp_im = rgb2gray(temp_im)
            temp_im = img_as_float32(temp_im)
            if self.pre_proc_transforms:
                temp_im = self.pre_proc_transforms(temp_im)
            self.files.append(samp_file)
            self.data.append([temp_im, 0])

        if self.bootstrap_template or self.positives_path:
            n_positives = len(self.class_2_index) - 1
            if self.verbose:
                print(f"Found {n_positives} examples...")
            for f in os.listdir(self.positives_path):

                templ_name = os.path.basename(f).split(".")[0]

                samp_file = os.path.join(self.positives_path, f)
                template_im = io.imread(samp_file)
                template_im = rgb2gray(template_im)
                template_im = img_as_float32(template_im)

                if self.pre_proc_transforms:
                    template_im = self.pre_proc_transforms(template_im)

                if self.bootstrap_template:
                    for j in tqdm(
                        range(len(self.files)),
                        disable=self.verbose == False,
                        desc="Augmented Template Creation:",
                    ):
                        self.data.append(
                            [template_im, self.class_2_index.get(templ_name)]
                        )
                else:
                    self.data.append([template_im, self.class_2_index.get(templ_name)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, text = self.data[index]
        if self.transform:
            img = self.transform(img)

        return img, text


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"],))
