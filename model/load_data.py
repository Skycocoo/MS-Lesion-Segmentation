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
            image = nib.load(self.model[i])
            segment = nib.load(self.seg[i])
            # self.data[image.shape][i][0]: image
            # self.data[image.shape][i][1]: segment
            self.data[image.shape].append([image.get_fdata(),
                                           segment.get_fdata()])

    def show_sample(self):
        def show_slices(slices):
            fig, axes = plt.subplots(1, len(slices), figsize=(10, 10))
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.T, cmap="gray", origin="lower")

        shape, sample = random.choice(list(self.data.items()))
        show_slices([sample[0][108, :, :],
                    sample[0][:, 230, :],
                    sample[0][:, :, 230]])
        show_slices([sample[1][108, :, :],
                    sample[1][:, 230, :],
                    sample[1][:, :, 230]])

    def generator(self):
        # need shuffle, validation?
        for i in self.data:
            input = []
            target = []
            for j in range(len(d.data[i])):
                # print(d.data[i][j][0].shape, d.data[i][j][1].shape)
                input.append(data[i][j][0])
                output.append(data[i][j][1])
            yield input, output
