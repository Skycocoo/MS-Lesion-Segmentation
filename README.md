## Notes

### Data

https://portal.fli-iam.irisa.fr/msseg-challenge/data

### File details

#### ./

- [useful] MS_segmentation.ipynb: actual training code (for testing purpose)
- [useful] Post_processing.ipynb: reconstruct patches into original images, and store the results

- Test_reconstruct.ipynb: previous Reconstruct file (extracted to class)
- Test_weight_merge.ipynb: test the idea of weighted reconstruct
- Test_weight_patch.ipynb: test original reconstruct algo
- Demonstrate_result.ipynb: plot training results vs ground truth on testing data
- Format_report.ipynb: format the structure of the network for latex report
- Test_data.ipynb: test the shape of data


- Training.py: Training with Dice Loss
- Training-binary.py: Training with Binary Cross Entropy
- Training-all.py: Training with all data (not selected)

#### ./model/

- data.py: fetch data into h5 format, preprocess data for training (padding, patch index, etc.), load data from h5 file
- dice.py: dice score calculation function
- generator.py: generate patches for training (different for training / testing dataset)
- model.py: 3D U-Net model structure setup (Keras)
- recon.py: reconstruct patches to original image shape (each instance: corresponds to one original image shape); 5 K-fold, 15 images, testing images: 3 (or else storing need additional counter)
