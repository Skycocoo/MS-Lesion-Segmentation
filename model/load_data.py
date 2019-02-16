import glob, os
import nibabel as nib
import matplotlib.pyplot as plt

# directory: ./data/*/*.nii.gz
# there are different modalities that should be taken care of


class Data:
    def __init__(self):
        self.model = []
        self.seg = []
        self.model_data = []
        self.seg_data = []

    def fetch_file(self):
        root, sub_dir, _ = next(os.walk(os.getcwd() + '/data/'))
        for sub in sub_dir:
            self.model.append(os.path.join(root, sub + '/FLAIR_preprocessed.nii.gz'))
            self.seg.append(os.path.join(root, sub + '/Consensus.nii.gz'))

    def load_data(self):
        self.fetch_file()
        for i in range(len(self.model_data)):
            self.model_data.append(nib.load(self.model[i]).get_fdata())
            self.seg_data.append(nib.load(self.seg[i]).get_fdata())

    def show_sample(self, files):
        def show_slices(slices):
            fig, axes = plt.subplots(1, len(slices), figsize=(10, 10))
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.T, cmap="gray", origin="lower")

        for f in files:
            img = nib.load(f)
            img_data = img.get_fdata()
            show_slices([img_data[108, :, :], img_data[:, 230, :], img_data[:, :, 230]])
