import tensorflow as tf
import numpy as np

### data loader 구현
class ReviewCoverLoader(tf.keras.utils.Sequence):
    def __init__(self, number, review, imageDict, objects, y_label, InputSize_wh, batch_size, shuffle = True,outputShape = None):
        self.number = number
        self.review = review
        self.imageDict = imageDict
        self.objects = objects
        self.y_label = y_label
        self.InputSize_wh = InputSize_wh
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.outputShape = outputShape
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.review.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(self.review.shape[0] / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        review = self.review[indexes]
        objects = self.objects[indexes]
        y = self.y_label[indexes]
        number = self.number[indexes]

        images = np.zeros(shape=(len(number), self.InputSize_wh, self.InputSize_wh, 3),dtype=np.float)
    
        for i in range(len(number)):
            images[i] = self.imageDict[i]

        return [review, images, objects],y