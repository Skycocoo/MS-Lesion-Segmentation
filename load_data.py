import glob, os
import nibabel as nib
import matplotlib.pyplot as plt

# directory: ./data/*/*.nii.gz

def fetch_data():
    model = []
    seg = []
    root, sub_dir, _ = next(os.walk(os.getcwd() + '/data/'))
    for sub in sub_dir:
        model.append(os.path.join(root, sub + '/FLAIR_preprocessed.nii.gz'))
        seg.append(os.path.join(root, sub + '/Consensus.nii.gz'))
    return model, seg

def show_sample(files):
    def show_slices(slices):
        fig, axes = plt.subplots(1, len(slices), figsize=(10,10))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
    for f in files:
        img = nib.load(f)
        img_data = img.get_fdata()
        show_slices([img_data[108, :, :], img_data[:, 230, :], img_data[:, :, 230]])
