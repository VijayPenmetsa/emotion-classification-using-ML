# Emotion Classification using ML

## Introduction

## Files
**emotion_classification.ipynb** contains all project code.

**training.csv and testing.csv** contains the data.

## Dataset

**Shape of the dataset:**
x_training - (28709, 48, 48),   y_training - (28709, 7),   x_testing - (3589, 48, 48),   y_testing - (3589, 7).

We have images of 48x48 resolution with 7 different classes (angry, disgust, fear, happy, sad, surprise, neutral)

<img width="249" alt="Screen Shot 2022-04-15 at 2 12 46 AM" src="https://user-images.githubusercontent.com/50517893/163529156-74ab0dd8-a801-42ef-9c54-57f74a181ba6.png">

## Modeling

### Binary Classification for happy and sad faces.

- So initially we will categorize faces as happy or sad.
- For this we will sperate happy and sad data from rest of the emotions.
- Shape of the resulting dataset is 13593 images with 48x48 resolution.

###  Principal Component Analysis (P.C.A)
- We use PCA to reduce the dimensionality of the dataset. In this case we compress our image data.
- We have fit PCA by changing the components parameters from 2304 to 256.
- The result is PCA was able to achieve 95% variance using only 256 components as opposed to 2304 components

<img width="387" alt="Screen Shot 2022-04-15 at 2 20 13 AM" src="https://user-images.githubusercontent.com/50517893/163529902-867384d8-9f13-47f7-8915-225482308442.png">

<img width="382" alt="Screen Shot 2022-04-15 at 2 20 33 AM" src="https://user-images.githubusercontent.com/50517893/163529936-90a58e4d-94b1-4f9a-93ef-1ddf61f746da.png">

Result is almost 90% memory saving. This will help the models train much faster.

###  linear discriminant analysis (L.D.A)

let's try to fit LDA on the dataset without PCA

<img width="734" alt="Screen Shot 2022-04-15 at 2 25 22 AM" src="https://user-images.githubusercontent.com/50517893/163530433-1ab95972-72b7-4075-9878-ac1e7338ee44.png">

### Binary Classification with Naive Bayes, Logistic Regression, SVM

<img width="729" alt="Screen Shot 2022-04-15 at 2 27 32 AM" src="https://user-images.githubusercontent.com/50517893/163530615-3cf0ac86-f969-4631-94ee-59e790ad4f44.png">

<img width="692" alt="Screen Shot 2022-04-15 at 2 27 45 AM" src="https://user-images.githubusercontent.com/50517893/163530633-2847f2cd-b121-4962-8afa-cf4b652b1994.png">

## Binary classification analysis

As a standalone model LDA did a decent job of 64.84% accuracy, but overall PCA achieved better results when combined with Logistic Regression and SVM models with an accuracy over 68% and 74% respectively.
Training time with PCA was significantly reduced because of the lesser number of components in the images.
So we decided to go with PCA for rest of the job.

## Multi-class Classification
Now, instead of classifying only happy and sad faces, we will consider all the 7 classes.

**Variance curves of all emotions data and happysad data after applying PCA:**

As we can see in the following figure, Both the curves are pretty much overlapping.

<img width="380" alt="Screen Shot 2022-04-15 at 2 32 48 AM" src="https://user-images.githubusercontent.com/50517893/163531125-595e2004-857e-420c-bf49-19c6f36ecb57.png">

**After and Before PCA**

<img width="326" alt="Screen Shot 2022-04-15 at 2 33 35 AM" src="https://user-images.githubusercontent.com/50517893/163531180-c7b43b4f-4dd0-4d87-ad80-c958f4ba19e9.png">


## LDA on all emotions dataset

LDA's accuracy on multi-class is 36.70

## Accuracies and analysis for other models on multi-class dataset

Accuracy (Naive Bayes):  24.84708267098717

From the figure below, we can see that Naive Bayes has the least accuracy of 24%, and most of the accuracy is coming from 3 classes 1, 2, and 3, which are disgust, fear, and happy.

Naive Bayes also got the least accuracy with binary classification but atleast it gave acceptable performance compared to multi-class classification.

Confusion Matrix:

<img width="356" alt="Screen Shot 2022-04-15 at 2 41 07 AM" src="https://user-images.githubusercontent.com/50517893/163531991-cc79d83c-5d8e-4110-a549-a37b92ec727a.png">


Accuracy (Logistic Regression):  35.467151774688254

Accuracy for logistic regression on multi-class is 35% which is better than Naive Bayes, but still not acceptable.
From the figure below we can see that the model's highest performance comes from class 3, and 5, which are happy and surprise. The performance is not bad with over 50% but overall when we consider all the classes, the model performed poorly.

Confusion Matrix:

<img width="353" alt="Screen Shot 2022-04-15 at 2 41 32 AM" src="https://user-images.githubusercontent.com/50517893/163532030-7c3836c0-1431-45e9-8e5b-47edcb81dc7d.png">



Accuracy (SVM):  42.86321086245709

<img width="355" alt="Screen Shot 2022-04-15 at 2 41 51 AM" src="https://user-images.githubusercontent.com/50517893/163532073-37afc603-76eb-4d85-ac9b-9944459f3112.png">

Out of all the models svm has the highest accuracy of 42%, but only one class has good performance which is 3 (happy). 4, 5, and 6 classes were a bit more consistent compared to the previous models, but 0, 1, and 2 did poorly.

Confusion Matrix:

<img width="359" alt="Screen Shot 2022-04-15 at 2 45 54 AM" src="https://user-images.githubusercontent.com/50517893/163532517-c56e8988-462d-4440-b4e5-50fe0437952c.png">

Based on the confusion matrix:
LEAST SIMILAR:
For SVM and Logistic regression the (1) disgust faces were the least similar to other classes.
But for naive bayes (5) surprise is the least similar to all classes.
HIGHLY SIMILAR:
For SVM and Logistic regression the (3) happy faces were the most similar faces to other classes.
But for naive bayes (2) fear was most similar to other classes.







