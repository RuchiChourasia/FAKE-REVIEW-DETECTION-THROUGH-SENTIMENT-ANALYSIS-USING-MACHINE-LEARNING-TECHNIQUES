# FAKE-REVIEW-DETECTION-THROUGH-SENTIMENT-ANALYSIS-USING-MACHINE-LEARNING-TECHNIQUES

Worked with a recently released corpus of Amazon reviews. 
A SVM model is used that classifies the reviews as REAL or FAKE. Used both the review text and the additional features contained in the data set to build a model that predicted with over 80% accuracy.

Step-By-Step Explanation of Detection of Fake Review.

1. Load the Dataset.
2. Shuffle the Dataset.
3. Changed the Label from __label1__ to FAKE and __label2__ to REAL.
4. Splitting Dataset for Training and Testing.

Training the Model using Train Dataset:
1. Parsing the datas from the Train set which are required.
2. Pre-processing the Text Data in the Train set by removing Stop-words and punctuations and Lemmatizing the tokens. 
3. Calculating TF-IDF for each document available in the Train Set. For Calculation of TF-IDF value:
  a. Calculate TF (Term Frequency) of each document. Which is counting of each token appears how many times in that document divides to the      length of the document and maintain a global dictionary of all the token which appears in all the documents with there counts.
  b. Calculate IDF (Inverse Document Frequency) which is calculated using the whole training set document count and the global dictionary.
     IDF = log( Total Number of documnents / occurrences of that term in all the documents )
  c. Calculate TF-IDF which is multiply both TF and IDF term for the document.
4. This TF-IDF output is used as input for the Linear SVM classifier. 
   (Here, Label is also added as for training we require both Data and corresponding Label)
   
Testing the Model using Test Dataset:
1. Do all the parsing and pre-procesiing for the Test set which is done for the training set as well.
2. Compute the TF-IDF values for the Test set.
3. Now, predict the labels for documents of Test set.
4. Evaluate the difference between the true label and the predicted labels.

Used Accuracy, Precision, Recall and F1-Score as the evaluation metrics for the model.

Cross-Validation is used to improve the accuracy of the Model. Here, folds is taken as 10.

Same approach is used for Bigram tokenization of the text. Above is done for unigram model.

Also, Term Occurences is used rather than TF-IDF computation. With both Unigram and Bigram model. 

Files contained in the folder FAKE_REVIEW_DETECTION_THROUGH_SENTIMENT_ANALYSIS_USING_MACHINE_LEARNING_TECHNIQUES :

TF_IDF_SVM : Contains the code of TF-IDF with SVM classifier. (Used both the review text and the additional features contained in the                 data set )

TF_IDF_SVM_WITH_BIGRAM : Contains the code of TF-IDF with SVM classifier with Bigram tokens as input. (Used both the review text and the                          additional features contained in the data set )

TO_SVM : Contains the code of Term occurrences with SVM classifier.

TO_SVM_BIGRAM : Contains the code of Term occurrences with SVM classifier with Bigram tokens as input.
