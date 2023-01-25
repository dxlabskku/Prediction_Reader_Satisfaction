# Can book covers help predict bestsellers using machine learning approaches?
## Abstract
* As the book publishing market changes from offline to online, readers tend to purchase books while paying more attention to book covers and metadata rather than the actual book contents. We examine whether publishers can know users’ satisfaction with
books in advance, and both metadata and book covers help predict this satisfaction.
* Exploring effects of metadata and book covers on the satisfaction is not only necessary for publishers’ perspectives, but also for librarians’ perceptions. However, the majority of prior research on user preference-based book recommendation systems in both book
industry and library system employed review comments, ratings, or book loan records.
* Thus, we open up the potentiality of other factors, which implicitly affect the satisfaction with books. We collected book titles, authors, publishers, reviews, ratings, and covers from the “Literature and Fiction” genre in the Amazon bookstore and conducted
an experiment to predict readers’ satisfaction ratings based on book reviews, metadata, and book covers. Several deep learning classifiers (CNN, ResNet, LSTM, BiLSTM, GRU, BiGRU) were employed.
* Reviews alone can reach a certain level of prediction performance, but adding metadata, cover images, and cover objects to a review-based predictive model slightly improves that performance. Based on these results, we confirmed that both metadata and book covers improve predicting readers’ perceived
satisfaction.
* This study is a pilot exploration of the idea that multimodal approaches can improve the prediction of the perceived satisfaction of book readers.
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
<img src="https://user-images.githubusercontent.com/42277033/214619252-bd5f1d94-64bb-4209-92e4-b6e632539ced.png"  width="800" height="320"/>

In terms of best accuracy and average accuracy, adding metadata and cover images to a review-based predictive model slightly improved performance, but adding cover objects reduced performance.
## Dataset
We put 100 sample data for testing in the 'test_data' folder. The 'test_data' folder contains raw data and preprocessed data. 'test.ipynb' is the file for testing.
