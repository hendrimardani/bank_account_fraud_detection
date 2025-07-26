# Conclusion
- In the dataset used, which contains 1 million rows of data, there is a difference in the amount of data 
between positive and negative classes in the target variable, with the negative class having more 
data than the positive class. This difference has a ratio of 90:10, so 
a slicing technique was applied to the negative class to balance the amount of data. The researcher 
did not use the SMOTE technique because the ratio difference was 
quite significant. The concept of the SMOTE technique involves replicating a specific class, 
which means duplicating the data, and this could lead to 
the model becoming overfitted.
- The feature extraction technique used is SelectKBest with the 5 best features, 
including month, velocity_4w, velocity_24h, housing_status, and 
credit_risk_score. There are outliers in credit_risk_score and housing status, 
so outlier removal is performed on these features.
- Deep learning methods are superior to classical machine learning because 
deep learning methods are generally used for complex tasks, ranging from 
structured data to unstructured data.


For recommendations from researchers, it is suggested to balance the amount of 
uniform data in each target variable class, namely the fraud_bool feature, to 
obtain  optimal  results  and  reduce  the  occurrence  of  model overfitting.  Due to 
the scope and limitations of the research, it is recommended to perform hyperparameter 
tuning again on the deep learning method to achieve more optimal results. Hyperparameter 
tuning can be performed on the number of layers, neurons, and activation function settings
such as ReLu and so on.
