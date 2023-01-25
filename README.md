# Can book covers help to know bestsellers?
## Abstract
* As the book publishing market changes from offline to online, readers tend to purchase books paying more attention to the book cover and metadata rather than the actual contents of the book. However, the previous studies that predicted reader satisfaction so far were mainly based on review comments. Therefore, we conducted a study for predicting reader satisfaction based on reviews, metadata, and book covers. In particular, we aimed to investigate whether the prediction performance can improve, when metadata and book covers are added to the review-based prediction model.
* We collected title, author, publisher, reviews, ratings, and book cover for ‘Literature and Fiction’ genre books in Amazon bookstore, and conducted an experiment to predict multi-rating based on review, metadata, and book cover. For this, several deep learning classifiers (CNN, ResNet, LSTM, BiLSTM, GRU, BiGRU) were employed.
* Reviews alone can reach a certain level of performance, and adding metadata and cover images to a review-based predictive model slightly improved performance. Through these results, we can confirm that metadata and cover images help predict reader satisfaction, but their effect is insignificant.
* This study is meaningful in that it is a study based on multimodal data in which image data is added to text data, and showed that performance can improve a bit by adding image data to the existing book rating prediction models centered on text data.
## Data Collection
<img src="https://user-images.githubusercontent.com/42277033/150510126-84e632b4-5ecd-43e2-9914-a2f3e53f2852.jpg"  width="500" height="200"/>

## Methods – classification models
* We implemented four case models according to the input data:
1) models with book reviews
2) models with book reviews and metadata
3) models with book reviews and metadata, and cover images
4) models with book reviews, metadata, cover images, and cover objects
* We used CNN, LSTM, BiLSTM, GRU, and BiGRU for review and metadata, CNN, ResNet for cover images, and DNN for cover objects.
## Architecture of the fused deep learning model
![architecture](https://user-images.githubusercontent.com/42277033/150513533-5296ff0f-675c-4db2-95f0-70be1813065b.jpg)
(Example of CNN+LSTM+ DNN)
## Results
#### 1. Performance comparison of machine learning and deep learning models based on book reviews
<img src="https://user-images.githubusercontent.com/42277033/150513894-052bfb01-e3d8-4dcd-8caf-e8ae5a2104dd.png"  width="800" height="200"/>

The best accuracy of the deep learning models is higher than that of machine learning models.

#### 2. Performance comparison of the baseline model and all other improved models
<img src="https://user-images.githubusercontent.com/42277033/150514523-7056a74e-33dc-426d-bcf7-7988bbca2a26.png"  width="800" height="320"/>

In terms of best accuracy and average accuracy, adding metadata and cover images to a review-based predictive model slightly improved performance, but adding cover objects reduced performance.
## Dataset
We put 100 sample data for testing in the 'test_data' folder. The 'test_data' folder contains raw data and preprocessed data. 'test.ipynb' is the file for testing.
