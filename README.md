## Notes

- 1208110: binary cross entropy
- 1215762: merge image with softmax weight
- 1216205: dice, train on all patches (with / without lesion)


## Todo

- V-net (3D) / U-net (2D)
- classify / cluster lesion (subtypes)

---

- soft dice loss / hard dice loss
- find more data to train

---

- modify Data to fit keras [DataGenerator](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
- modify sbatch file for more training time
- merge image directly instead of storing all patches first


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
