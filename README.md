# cs229stockprediction
Novel approaches to sentiment analysis for stock prediction   
Chris Wang, Yilun Xu, Qingyang Wang   
Stanford University   
{chrwang, ylxu, iriswang} @ stanford.edu  

## Introduction
Stock market predictions lend themselves well to a machine learning framework due to their quantitative nature. A supervised learning model to predict stock movement direction can combine technical information and qualitative sentiment through news, encoded into fixed length real vectors. We attempt a large range of models, both to encode qualitative sentiment information into features, and to make a final up or down prediction on the direction of a particular stock given encoded news and technical features. We find that a Universal Sentence Encoder, combined with SVMs, achieve encouraging results on our data. 
![Optional Text](../master/src/combinedmodel.png)

## Requires
- scikit-learn 
- pytorch
- keras
- tensorflow
- tensorflow hub `pip install tensorflow-hub`

### Use the following files for data gathering and preprocessing
- GoogleNewsScraper.py
- NYtimesScraper.py
- Preprocessing.py

### Use the following file for text representation
- GoogleUSE_PCA.py

### Use the following file for Stock Movement Prediction
- Logreg_SVM.py
- NeuralNetwork.py
- RNN.py
- CNN.py
