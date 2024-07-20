# Machine Learning

Have you ever noticed that if you watch any video on youtube, it automatically creates a related `playlist` sideby:

![image](https://github.com/user-attachments/assets/cbce6ac7-ec27-4ead-8828-18b76d7f2c1b)

When you search a product on shopping website, it shows a `Bought together` column:

![image](https://github.com/user-attachments/assets/56f08c13-a7bb-490e-8d0f-3d1dd056299d)

You also have observed the `SPAM` section of your email, how come some suspected emails automatically move to spam section:
It's all because of machine learning.

# What is Machine Learning

Machine learning is to make computer learn and act like humans. 
A ML system learns from historical data, builds the prediction models, and whenever recieves new data, predicts the output for it.

![image](https://github.com/user-attachments/assets/afb73e53-9a16-42d8-b884-fec096163b0a)

# Traditional Programming vs Machine Learning

![image](https://github.com/user-attachments/assets/cd94c20f-dfd7-497e-882d-1bb8cacf2b0a)

Example: We have Inputs and Output:

|Input 1|Input 2|Output|
|-------|-------|------|
|2|1|2|
|2|2|4|
|2|3|6|
|2|10|?|

In Traditional programming, we will provide:
<br>**Data(Input):** Input 1, Input 2
<br>**Program:** Input 1 * Input 2
<br>and get: **Output**

In Machine Learning, we will provide:
<br>**Data(Input):** Input 1, Input 2
<br>**Output:** Output
<br>and get **Program** and using the program genrate outputs from new data.

# Classifications of Machine Learning

- **Supervised Machine Learning**
    - For future prediction or making a recommendation system
    - Example, a real estate agent predicting the price of a house based on locality and infrastructure of neighbouring houses.
- **Unsupervised Machine Learning**
    - Involves groupism or classification by detecting a pattern.
    - Take a bucket of mixed variety of fruits and then create batches of similar type of fruits and put them under a specific category. Similarly, machine looks for a specific pattern in data and create groups of similar pattern data.
- **Reinforcement Machine Learning**
    - Automatic game play
    - In Ludo King, when playing computer vs you, computer uses reinforcement learning after your every move to decide next move.

# Advantages of ML

- Easily detects patterns and trends and helps in prediction.
- Reduces human intervention(Automation). Example, youtube automatically creates a playlist of your frequently watched videos.
- Handling multi-variety of bulk data.

# Disadvantages of ML

- Data acquisition as large amount of data need to be collected for training an ML model.
- Large amount of Time and Resources required for cleaning, processing and filtering data.

# Uses of ML

![image](https://github.com/user-attachments/assets/7a36d3da-28d3-4b6e-91a2-37fbed8d9a14)

# ML Roadmap

![image](https://github.com/user-attachments/assets/4523eb2d-789b-4bf1-87cf-2ec009ba11e9)

# Python Libraries for ML

- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scipy
- Scikit-learn
- Tensorflow (Deep Learning)
- Nltk
- OpenCV

# Types of Variables/Data in ML

![image](https://github.com/user-attachments/assets/8d01a372-226c-461d-b092-489806e4dbbb)

![image](https://github.com/user-attachments/assets/f638b161-e548-4391-9fe5-e8a23a35982a)

Other variables:
- Data and Time variables
- Mixed variables(both number and categorical data)

# Data cleaning

We saw that there could be different types of data available for training an ML model, but we can use only numerical type of data for training, so we need to do data cleaning to convert our entire data into numeric type.

Data cleaning is the process of preparing data for analysis, ML, DL by removing or modifying data that is incorrect, irrelevant, incomplete, duplicated, or improperly formatted.

![image](https://github.com/user-attachments/assets/e6ed86f7-8de5-4062-84fb-51237a68d689)

### 1. Handling Missing Data
Missing data can occur due to various reasons such as human error, equipment malfunction, or data corruption. It's important to handle missing data because it can lead to biased or inaccurate results as ML models are generally Maths formula. Common techniques include:
- **Removing**: If the amount of missing data is small and seems random, you can remove those rows or columns.
- **Imputing**: Replace missing values with a substitute value, such as the mean, median, mode, or a value estimated using regression models.
- **Using Algorithms that Support Missing Values**: Some algorithms can handle missing data internally.

![image](https://github.com/user-attachments/assets/abefab8e-30c1-45a2-b007-431e227f162a)


### 2. Outlier Detection and Handling
Outliers are data points that are significantly different from the majority of the data. They can skew and mislead the training process of a machine learning model. Methods to handle outliers include:
- **Statistical Methods**: Using z-scores or the IQR (Interquartile Range) method to identify and remove outliers.
- **Visual Methods**: Using plots like box plots or scatter plots to visually detect outliers.
- **Capping**: Limiting extreme values to a certain percentile.
- **Transformation**: Applying transformations like log, square root, or Box-Cox to reduce the impact of outliers.

### 3. Data Scaling and Transformation
Data scaling and transformation are crucial when the features of your dataset have different ranges or units. This ensures that each feature contributes equally to the model. Common techniques include:
- **Standardization**: Transforming data to have a mean of zero and a standard deviation of one.
- **Normalization**: Scaling data to a range of [0, 1] or [-1, 1].
- **Log Transformation**: Reducing skewness by applying a logarithmic function.
- **Power Transformation**: Stabilizing variance and making the data more normally distributed using functions like Box-Cox.

### 4. Encoding Categorical Variables
Categorical variables need to be converted into a numerical format since most ML algorithms require numerical input. Common encoding techniques include:
- **Label Encoding**: Assigning each category a unique integer.
- **One-Hot Encoding**: Creating binary columns for each category.
- **Ordinal Encoding**: Similar to label encoding but respecting the ordinal nature of the categories.
- **Target Encoding**: Encoding categorical variables based on the target variable.

### 5. Handling Duplicates
Duplicate data can lead to biases and inefficiencies in the model. Steps to handle duplicates include:
- **Identifying Duplicates**: Using methods like `df.duplicated()` in pandas.
- **Removing Duplicates**: Dropping duplicate rows using `df.drop_duplicates()`.
- **Deciding Based on Context**: Sometimes duplicates are valid and should be retained based on domain knowledge.


### 6. Dealing with Inconsistent Data
Inconsistent data can arise from various sources such as different formats, misspellings, or variations in case. Handling inconsistent data involves:
- **Standardizing Formats**: Converting data to a consistent format (e.g., date formats).
- **Correcting Misspellings**: Using dictionaries or algorithms to correct typos.
- **Consistent Casing**: Converting text data to a consistent case (e.g., all lowercase).
- **Using Domain Knowledge**: Applying specific rules based on the context of the data to resolve inconsistencies.

By thoroughly cleaning data through these steps, you can ensure that your dataset is accurate, reliable, and ready for effective analysis or modeling.


# Finding Missing Data

![image](https://github.com/user-attachments/assets/5736ec1a-99ce-41bb-8b07-762a5340c8f5)

![image](https://github.com/user-attachments/assets/a1a1dc55-bf11-4afb-a47b-932b55e0ba7b)

![image](https://github.com/user-attachments/assets/91cd551f-4cff-4bcf-b2d0-c32606e724d5)

![image](https://github.com/user-attachments/assets/e3119541-55e6-47dc-b7f2-cadee806637f)

![image](https://github.com/user-attachments/assets/db69172a-0a3d-452e-b93e-6e77d5607545)

![image](https://github.com/user-attachments/assets/4e78a400-4db7-44c9-a621-c23bb1a17fb4)

![image](https://github.com/user-attachments/assets/3609ae3a-17c1-48fa-a925-13e2fbafdba7)

![image](https://github.com/user-attachments/assets/4429d86e-c8db-4e4b-a737-e3022df1553d)

# Handling Missing Data

You need thorough knowledge of your data set. Don't use a dataset with 50%+ missing data.

![image](https://github.com/user-attachments/assets/7f248a80-df50-41b1-8955-5eee73d29a48)

![image](https://github.com/user-attachments/assets/1a60910b-c3e6-42c6-a0b2-7a50218cd9ac)

![image](https://github.com/user-attachments/assets/572dc9db-7c38-43eb-8ae7-5915fd01790c)

![image](https://github.com/user-attachments/assets/5ca1c8cf-c0db-400e-a1bb-a3b87424a1dc)

![image](https://github.com/user-attachments/assets/b6259e25-7fa7-4180-a064-d0afa39c00a5)

![image](https://github.com/user-attachments/assets/c607329e-38dc-4334-9e77-349caa323a41)

# Handling Missing Data(Imputing Categorical Values)

![image](https://github.com/user-attachments/assets/f1d62059-a43c-44fa-aeed-43b570cff0aa)
`Note:` This is wrong approach.

**Right approach:**

![Screenshot from 2024-07-20 12-23-04](https://github.com/user-attachments/assets/664d4725-9cd7-40c8-b5e8-abc7fdd7805c)


![Screenshot from 2024-07-20 12-29-48](https://github.com/user-attachments/assets/d12b49fd-c224-4839-a347-dadd6d895035)


# Handling Missing Data using Scikit Learn(Used in ML pipeline)

![Screenshot from 2024-07-20 12-39-00](https://github.com/user-attachments/assets/6a21fac5-25cf-47bd-b109-4df33bb5bcb2)

Sure! Let's delve deeper into supervised learning, its steps, and the algorithms used for classification and regression with examples.

# Supervised Learning

Supervised learning is a type of machine learning where the model is trained on labeled data. This means that each training example is paired with an output label. The goal is to learn a mapping from inputs to outputs and make accurate predictions for unseen data.

### Example: Email Spam Detection

#### Step 1: Data Collection
Collect a dataset of emails labeled as "spam" or "not spam." Each email in the dataset should have features such as the subject line, email body, sender, and other relevant information.

#### Step 2: Training
Use the labeled dataset to train a model. The model will learn patterns and characteristics of spam and non-spam emails from the training data.

#### Step 3: Testing
Test the trained model on a separate dataset of emails to evaluate its performance. The goal is to see how well the model can classify new, unseen emails as spam or not spam.

## Algorithms in Supervised Learning

- **Classification goal** is to assign input data to predefined categories or classes. Algorithms like K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, Support Vector Machines, and Neural Networks are commonly used for classification tasks.
**Example Scenario:** Classifying emails as "spam" or "not spam."

    - **Input Data:** Features extracted from emails, such as the presence of certain keywords, frequency of specific terms, email length, sender's address, etc.
    - **Output Classes:** "Spam" or "Not Spam."
 
      
- **Regression goal** is to predict a numerical value based on input features. Algorithms like Linear Regression, Polynomial Regression, and various types of regression trees are used for regression tasks. just give example for both
**Example Scenario:** Predicting house prices based on various features.

    - **Input Data:** Features of houses, such as square footage, number of bedrooms, number of bathrooms, location, age of the house, etc.
    - **Output Value:** The predicted price of the house.

