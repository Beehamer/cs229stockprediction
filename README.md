# cs229stockprediction
Novel approaches to sentiment analysis for stock prediction   
Chris Wang, Yilun Xu, Qingyang Wang   
Stanford University   

## Introduction
Stock market predictions lend themselves well to a machine learning framework due to their quantitative nature. A supervised learning model to predict stock movement direction can combine technical information and qualitative sentiment through news, encoded into fixed length real vectors. We attempt a large range of models, both to encode qualitative sentiment information into features, and to make a final up or down prediction on the direction of a particular stock given encoded news and technical features. We find that a Universal Sentence Encoder, combined with SVMs, achieve encouraging results on our data. 
![Optional Text](../master/src/combinedmodel.png)

## Requires
- scikit-learn `pip install sklearn`
- pytorch `pip install pytorch`
- keras `pip install keras`
- tensorflow `pip install tensorflow`
- tensorflow hub `pip install tensorflow-hub`

### Use the following files for data gathering and preprocessing
- GoogleNewsScraper.py
- NYtimesScraper.py
- Preprocessing.py

### Use the following file for text representation
- GoogleUSE_PCA.py

### Use the following files for Stock Movement Prediction
- Logreg_baseline.py (Logistic regression without sentiment data)
- Logreg_SVM.py (Logistic regression with sentiment data and SVM with rbf kernel on Google and NY times data)
- RandomForest.py (Random Forest on Google and NY times data)
- NeuralNetwork.py (Fully connected neutral network on Google and NY times data)
- RNN.py (RNN on Google and NY times data by ticker)
- CNN.py (CNN on Google and NY times data)

### Use the following file to output interactive prediction plot by ticker
- InteractivePlot.ipynb
