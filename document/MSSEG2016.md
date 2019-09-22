## [MICCAI 2016 Multiple Sclerosis Lesion Segmentation Challenge](https://www.nature.com/articles/s41598-018-31911-7)

- [data](https://portal.fli-iam.irisa.fr/msseg-challenge/data)
- [evaluation](https://github.com/Inria-Visages/Anima-Public)

## Approaches

[overview of all methods evaluated](https://portal.fli-iam.irisa.fr/msseg-challenge/workshop-day?p_p_id=110_INSTANCE_IQiAKumqNTIj&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-1&p_p_col_pos=1&p_p_col_count=2&_110_INSTANCE_IQiAKumqNTIj_struts_action=%2Fdocument_library_display%2Fview_file_entry&_110_INSTANCE_IQiAKumqNTIj_redirect=https%3A%2F%2Fportal.fli-iam.irisa.fr%3A443%2Fmsseg-challenge%2Fworkshop-day%3Fp_p_id%3D110_INSTANCE_IQiAKumqNTIj%26p_p_lifecycle%3D0%26p_p_state%3Dnormal%26p_p_mode%3Dview%26p_p_col_id%3Dcolumn-1%26p_p_col_pos%3D1%26p_p_col_count%3D2&_110_INSTANCE_IQiAKumqNTIj_fileEntryId=35910)

Teams with machine learning approaches: Team 5, 6, 9, 12; Team 6 and 12 uses convolutional neural network

### Team 1: (no machine learning): Multimodal Graph Cut

### Team 2: (no machine learning): Intensity-Normalized Multi-channel MRI

### Team 3: (no machine learning): P-LOCUS

### Team 4: (no machine learning): Edge-based lesion segmentation + fuzzy classification

### Team 5: Hybrid Artiﬁcial Neural Networks

Shallow neural network

Related challenge: MRBrainS13 challenge (2015)

#### Feature extraction

- Intensity of 3D FLAIR channel
- Intensity, gradient magnitude, and Laplacian of the intensity after convolution with Gaussian kernels with σ = 1, 2, 3 mm^2
- Spatial information of all voxels (x, y, z) which were divided by the length,
width, and height of the brain respectively

#### Network architecture

1 hidden layer, 100 hidden nodes, 50000 voxels randomly selected (90% brain tissue, 10% lesion), 1000 iterations, probability on classification of 2 classes (lesion or not)


### Team 6: Nabla-net: convolutional neural network on 2D images

[Nabla-net: a deep dag-like convolutional architecture for biomedical image segmentation: application to white- matter lesion segmentation in multiple sclerosis](https://link.springer.com/chapter/10.1007%2F978-3-319-55524-9_12)

Deep convolutional autoencoder to represent the distribution of lesion

#### Network architecture

(need screenshot)

Loss function: binary cross-entropy weighted by scaled intensity of lesion mask

#### Training

3 separate model trained on axial, saggital, and coronal data to be averaged to output the final result


### Team 7: (no machine learning): Random Forests

### Team 8: (no machine learning): Rules and level sets

### Team 9: Evaluation-Oriented Training Strategy

Shallow neural network

Evaluate based on 2D rules instead of 3D metrics (Dice similarity score, lesion-wise true positive rate, and lesion-wise positive predictive value)

#### Network architecture

Multilayer perceptron model, 1 hidden layer, few neurons (didn't mention how many neurons)

Loss: particle swarm optimization


### Team 10: (no machine learning): Spacial intensity distribution classification

### Team 11: (no machine learning): Max-tree representation of intensity

### Team 12: Convolutional neural network with 3D patches

[Improving automated multiple sclerosis lesion segmentation with a cascaded 3D convolutional neural network approach](https://www.sciencedirect.com/science/article/pii/S1053811917303270)

Shallow convolutional network

#### Network architecture

input -> convolution -> max pooling -> convolutioin -> max pooling -> dropout -> fully-connected -> softmax to 2 classes (lesion / not lesion)'

#### Training

Each 3D patch of 15 * 15 * 15 size for each image

2 separate models trianed on same positive lesion voxels and different negative voxels to represent different false positive regions

Batch size 4096, # epochs 50, adam optimization


### Team 13: (no machine learning) Random forest + Markov Random Field post processing



## Team Ranking

- Team 6 (convolutional autoencoder)
- Team 12 (convolutional neural network)

---

Segmentation Ranking: (Dice Score & Surface Distance)
-	Team 6	0.591
-	Team 8	0.572
-	Team 12	0.541
-	Team 13	0.521
-	Team 4	0.490
-	Team 2	0.485
-	Team 3	0.489
-	Team 1	0.453
-	Team 5	0.430
-	Team 11	0.347
-	Team 9	0.340
-	Team 7	0.341
-	Team 10	0.228

Lesion Detection Ranking: (F1 Score)

-	Team 12	0.490
-	Team 8	0.451
-	Team 13	0.410
-	Team 6	0.386
-	Team 3	0.360
-	Team 1	0.341
-	Team 2	0.294
-	Team 4	0.319
-	Team 11	0.188
-	Team 5	0.167
-	Team 9	0.168
-	Team 7	0.134
-	Team 10	0.049


### Other Resources

[Survey of automated multiple sclerosis lesion segmentation techniques on magnetic resonance imaging
Author links open overlay panel](https://www.sciencedirect.com/science/article/pii/S0895611118303227)
