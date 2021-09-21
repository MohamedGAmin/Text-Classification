# text-classification
Description: 

You can think of the job industry as the category or general field in which
you work. On a job application, "industry" refers to a broad category under
which a number of job titles can fall. For example, sales are an industry; job
titles under this category can include sales associate, sales manager,
manufacturing sales rep, pharmaceutical sales and so on.

Required:

Building a model using any Machine Learning classifier algorithm to classify job titles by the industry and providing insights on how the model works and create a RESTful API service where the input is a HTTP request with a parameter for the "Job title" and the output is the expected industry.

Cleaning the data:

•	There are a lot of duplicates in the “job title” column. So, I started with dropping the duplicates in the column which many of them were in “IT” class. Having data with a lot of duplicates might leads to false accuracy. 
•	I applied different text preprocessing techniques, such as: removing numerical values, removing symbols, adjusting spaces, removing words with one and two letters and removing stopping words. Unfortunately, these techniques did not improve the classifier. 

Encoding:

•	I used TF-IDF Vectorizer which weights the word counts by a measure of how often they appear in the data. I also tried Count Vectorizers which converts the words into frequency representation. TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the data but also provides the importance of the words. Higher value of TF-IDF signifies higher importance of the words in the data while lower values represent lower importance.

 
Dealing with Imbalanced data:

•	Imbalanced data are a common problem in machine learning classification where there is a disproportionate ratio of observations in each class. Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce error.   
•	There are different techniques to avoid imbalanced data: Changing the performance metric ‘from accuracy to recall or precision’, resampling ‘such as: oversample minority class and undersample majority class or adding class weights while training ‘The whole purpose is to penalize the misclassification made by the minority class by setting a higher-class weight and at the same time reducing weight for the majority class’.
•	After removing the duplicates from the data, I found that most of the rows removed were in the “IT” class which is the majority class. Afterwards I used compute_sample_weight from ‘sklearn’ which estimates sample weights by class for unbalanced datasets.

Model Selection:

•	After preprocessing the data, the first question I asked to myself “which machine learning classifier should I choose?”. This is a common asked question in the machine learning and the answer to this question depends on many factors. There are many algorithms I can use for this case. 
•	Firstly, I started with Naive Bayes Classifier as it’s easy to interpret with text data, however it may not perform well with sparse data and the multinomial variant is the most suitable variant in this case. 
•	Secondly, Linear Support Vector Machine as it’s widely regarded as one of the best text classification algorithms because it can solve linear and non-linear problems and work well for many practical problems as it creates a line or a hyperplane which separates the data into classes.
•	Thirdly, Logistic regression as it’s simple and easy to be generalized to multiple classes. 
•	Fourthly, Random Forest classifiers are suitable for dealing with the high dimensional noisy data in text classification. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
•	Finally, I chose Linear Support Vector Machine as it has the highest recall. 
 
Model Evaluation: 

•	It’s very important to choose the right metric to evaluate your model. As I mentioned earlier, accuracy not the best choice to be used as a measure when the target variable classes are imbalanced.
•	Accuracy is the number of correct predictions made by the model over all kinds predictions made.
•	Recall gives us information about a classifier’s performance with respect to false negatives (how many did we miss)
•	Precision gives us information about the classifier’s performance with respect to false positives (how many did we caught).
•	F1 score is the average between recall and precision, it might be a better measure to use if we need to seek a balance between Precision and Recall.
•	In this case, I focused more on recall as it calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive), in other words, it calculates how many our model predicted “IT” when the actual class was “IT”.

Limitations and improvements:

•	Data is very important to any model, the data used in this task was not large and not balanced enough.
•	Additional way to improve the model is to use pretrained models which could achieve higher accuracy for example using XLNet which perform well on major NLP tasks. 
•	Using deep learning techniques for text classification. 

Deploying the Model: 

•	Using Flask API to create a RESTful API for the final model. The model is pretrained and it will predict depends on the request given.
•	For requests: "http://127.0.0.1:5000/( job title )". 
•	For responses: it will be the predicted class by the model.

![image](https://user-images.githubusercontent.com/54632431/134147127-962d0778-c4d3-497e-8e4a-ed1c04755449.png)
