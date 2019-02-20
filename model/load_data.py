import glob, os, random
import nibabel as nib
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime


# directory: ./data/*/*.nii.gz
# there are different modalities that should be taken care of


class Data:
    def __init__(self):
        self.model = []
        self.seg = []
        self.data = defaultdict(list)
        self.valid_index = {}

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

    # need to zeropad image: shape divisible by pool ^ depth
    def zero_pad(self, image, pool_shape=(2, 2, 2), depth=4):
        pad_size = [0, 0, 0]
        pad = False
        for i in range(len(image.shape)):
            divident = exp(pool_shape[i], depth)
            remain = image.shape[i] % divident
            if remain != 0:
                pad = True
                # should be integer division
                div = image.shape[i] / divident
                pad_size[i] = (div+1) * divident - image.shape[i]
        if pad:
            # deal with odd number of padding
            pad0 = (pad_size[0]/2, pad_size[0] - pad_size[0]/2)
            pad1 = (pad_size[1]/2, pad_size[1] - pad_size[1]/2)
            pad2 = (pad_size[2]/2, pad_size[2] - pad_size[2]/2)
            # https://stackoverflow.com/questions/50008587/zero-padding-a-3d-numpy-array
            image = np.pad(image, (pad0, pad1, pad2), 'constant')

    def preprocess(self, pool_shape=(2, 2, 2), depth=4):
        # initialize validation index for training
        for i in self.data:
            self.valid_index[i] = None

        # preprocess training and validation data
        for i in self.data:
            for j in range(len(self.data[i])):
                self.zero_pad(self.data[i][j][0], pool_shape, depth)
                self.zero_pad(self.data[i][j][1], pool_shape, depth)
                # print(d.data[i][j][0].shape, d.data[i][j][1].shape)

    # K-fold LOOCV: leave one out cross validation
    def valid_index(self):
        for i in self.valid_index:
            random.seed(datetime.now())
            self.valid_index[i] = random.randint(0, len(self.data[i]))

    # batch_size: 2 or 4
    def train_generator(self, batch_size=2):
        for i in self.data:
            input = []
            target = []
            for j in range(len(self.data[i])):
                # skip validation data
                if j == self.valid_index[i]:
                    continue
                if len(input) < batch_size:
                    input.append(self.data[i][j][0])
                    output.append(self.data[i][j][1])
                else:
                    yield input, output
                    # reinitialize input and output
                    input = []
                    output = []

    # each scanner yield a simple validation sample
    def valid_generator(self):
        for i in self.valid_index:
            yield self.data[i][self.valid_index[i]][0], self.data[i][self.valid_index[i]][1]
