## Todo

- V-net (3D) / U-net (2D)
- Display data: [itk-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage)

- classify / cluster lesion (subtypes)
- cross-validation: 3:1

---

- load FLAIR as input and Consensus as training?
- deal with different input shape
- modify existing unet


## Notes

- [input with variable shape](https://github.com/keras-team/keras/issues/1920)

- [train on batches](https://github.com/keras-team/keras/issues/68)

```py
# each batch would be expected to have different sizes for MSSEG
# Alternatively, without data augmentation / normalization:
for e in range(nb_epoch):
    print("epoch %d" % e)
    for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
```
