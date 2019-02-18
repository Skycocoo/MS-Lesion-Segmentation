import glob, os
import nibabel as nib
import matplotlib.pyplot as plt
from collections import defaultdict

# directory: ./data/*/*.nii.gz
# there are different modalities that should be taken care of


class Data:
    def __init__(self):
        self.model = []
        self.seg = []
        self.data = defaultdict(list)

    def fetch_file(self):
        root, sub_dir, _ = next(os.walk(os.getcwd() + '/data/'))
        for sub in sub_dir:
            self.model.append(os.path.join(root, sub + '/FLAIR_preprocessed.nii.gz'))
            self.seg.append(os.path.join(root, sub + '/Consensus.nii.gz'))

    def load_data(self):
        self.fetch_file()
        for i in range(len(self.model)):
            # could load as pair of corresponding model-seg list?
            image = nib.load(self.model[i])
            segment = nib.load(self.seg[i])
            # self.data[image.shape][i][0]: image
            # self.data[image.shape][i][1]: segment
            self.data[image.shape].append([image.get_fdata(), segment.get_fdata()])

    def show_sample(self, files):
        def show_slices(slices):
            fig, axes = plt.subplots(1, len(slices), figsize=(10, 10))
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.T, cmap="gray", origin="lower")

        for f in files:
            img = nib.load(f)
            img_data = img.get_fdata()
            show_slices([img_data[108, :, :], img_data[:, 230, :], img_data[:, :, 230]])
