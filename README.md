# Alina-Portfolio
PROJECT 1- IMAGE CLASSIFICATION USING DEEP NEURAL NETWORK (PYTORCH)(https://github.com/alinashaji/Alina-Portfolio/blob/main/Image_classification_neural_network.ipynb)

In this model, we have created a 3 layer neural network for classifying the images in the FASHION MNIST dataset.(PYTORCH)

We used ReLU activation function to introduce non-linearity into the model, that helps our model to learn more complex relationships between the inputs (pixel densities) and outputs (class probabilities).

Used utilities like get_default_device, to_device and DeviceDataLoader which allows the model use GPU if availabe else use CPU

Our model was able to make fair predictions with an accuracy of 90%

Suggestion: Model can be improved by adding more hidden layers

PROJECT 2 - BREAST CANCER DETECTION https://github.com/alinashaji/Alina-Portfolio/blob/main/breast_cancer_detection_2_%20(2).pdf

Breast cancer is one of the most common causes of death among women worldwide. Early-stage detection would help the world to reduce the number of deaths. Breast cancer occurs when the cells begin to multiply abnormally and will result in a lump. The lumps present in the human body can be classified as cancerous and non-cancerous. Non-cancerous tumors are pretty common and they don’t have any symptoms.
UCI has shared a historical dataset that describes the diagnosis results of whether a tumor is malignant or benign. Malignant tumors are cancerous whereas Benign tumors are noncancerous.  
createD a fully automated system that uses the columns(characteristics) in the dataset to predict if a tumor in the breast is malignant or benign.




PROJECT 3 - MEDICAL INSURANCE PRICE PREDICTION(https://github.com/alinashaji/Alina-Portfolio/blob/main/insurance%20prediction.ipynb)
: 

Created a machine learning model for predicting the insurance price for the new customers by training the historical data.

Historical data set contains more than 1000's of unique entries

Did exploratory data analysis to establish correlation with inputs and target

 used one hot encoding, scaling numeric feature
 
 Used linear regression Algorithm for training the model 
 ![image](https://user-images.githubusercontent.com/101203819/159131983-8701d8fb-1fb5-4e0a-bf49-9464fd15a383.png)

 
 
 
 
 


 PROJECT 4 - WEATHER PREDICTION(https://github.com/alinashaji/Alina-Portfolio/blob/main/weather%20prediction.ipynb)

Commonwealth of Australia, Bureau of Meteorology has shared a historical dataset that contains about 10 years of daily weather observations from numerous Australian weather stations.

Created a model to predict the value of the column "Rain tomorrow" using the input values such as Date Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday.

Used One hot encoding, Imputing missing values, scaling numeric feature

Used Logistic regression algorithm for classification

created confusion matrix and obtained 85% accuracy for the model

![image](https://user-images.githubusercontent.com/101203819/159131945-22ed0d00-dced-4ca3-9205-8b321fd518da.png)









PROJECT 5 - MNIST HANDWRITTEN DIGITS IMAGE CLASSIFICATION(https://github.com/alinashaji/Alina-Portfolio/blob/main/Deep_learning_image_classification.ipynb)

MNIST dataset contain 28px by 28px grayscale images of handwritten digits from 0 to 9

Created a model using pytorch and logistic regression

Used softmax for generating output as probability

used evaluation metrics "accuracy" and loss function "cross entropy"

Trained the model and evaluated model using a validation set

Tested the model on randomly selected examples

![image](https://user-images.githubusercontent.com/101203819/159131837-50857614-be9d-4d92-9cb8-9ae3bc2e0dc3.png)
![image](https://user-images.githubusercontent.com/101203819/159131880-08435807-6ecb-4edd-a774-85aac93851b6.png)


