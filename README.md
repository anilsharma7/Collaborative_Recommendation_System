# Collaborative_Recommendation_System
A documentation on building collaborative filtering models for recommending products to customers

# **Build a Recommendation System for Grocery Purchase Data**

We will be using item-based **COLLABORATIVE FILTERING MODEL** as RECOMMENDATION SYSTEM with help of Python & ML module Turicreate

Recommendation technology is used to achieve **GOAL** *to personalize grocery products and offers to customers using purchase data.*

Below steps will be used to accomplish this goal:

* Transforming and normalizing data
* Training Models
* Evaluating Model Performance
* Selecting the optimal model

# **Product Overview**

Suppose a grocery chain release a new mobile app to allow its customers to place orders before they even walk into store.

There is an **OPPORTUNITY** for the app *to show recommendations*:
When a customer first taps on the "ORDER" page, company may recommend top 10 items to be added to their basket. e.g. Fresh vegetables, Salad, Chips, Meat, Cookies, Bread, Honey and so on.



# **Problem Statement**
In this data challenge, we are building collaborative filtering models for recommending product items. The steps below aim to recommend users their top 10 items to place into their basket. The final output will be a csv file in the output folder, and a function that searches for a recommendation list based on a speficied user:

   **Input**: user - customer ID

   **Output**: ranked list of items (product IDs), that the user is most likely to want to put in empty "basket"



# **Implementation**

# **Import modules**
* **pandas** and **numpy** for data manipulation
* **turicreate** for performing model selection and evaluation
* **sklearn** for splitting the data into train and test set

# **Load data**
Two datasets are used in this exercise, which can be found in *content* folder:

* *recommend_1.csv* consisting of a list of 1000 customer IDs to recommend as output

* *tran_data.csv* consisting of user transactions


# **Collaborative Filtering Model**
* In collaborative filtering, we would recommend items based on how similar users purchase items. For instance, if customer 1 and customer 2 bought similar items, e.g. 1 bought X, Y, Z and 2 bought X, Y, we would recommend an item Z to customer 2.

* To define similarity across users, we use the following steps:

  1. Create a **user-item matrix**, where index values represent unique customer IDs and column values represent unique product IDs

  2. Create an **item-to-item similarity matrix**. The idea is to calculate how similar a product is to another product. There are a number of ways of calculating this. In steps 6.1 and 6.2, we use cosine and pearson similarity measure, respectively.

    * To calculate similarity between products X and Y, look at all customers who have rated both these items. For example, both X and Y have been rated by customers 1 and 2.
    * We then create two item-vectors, v1 for item X and v2 for item Y, in the user-space of (1, 2) and then find the cosine or pearson angle/distance between these vectors. A zero angle or overlapping vectors with cosine value of 1 means total similarity (or per user, across all items, there is same rating) and an angle of 90 degree would mean cosine of 0 or no similarity.
  3. For each customer, we then **predict his likelihood to buy a product** (or his purchase counts) for products that he had not bought.

    * For our example, we will calculate rating for user 2 in the case of item Z (target item). To calculate this we weigh the just-calculated similarity-measure between the target item and other items that customer has already bought. The weighing factor is the purchase counts given by the user to items already bought by him.
    * We then scale this weighted sum with the sum of similarity-measures so that the calculated rating remains within a predefined limits. Thus, the predicted rating for item Z for user 2 would be calculated using similarity measures.
* We can use **turicreate** library to capture different measures like using **cosine** and **pearson** distance, and evaluate the best model.


# **Cosine similarity**
* Similarity is the cosine of the angle between the 2 vectors of the item vectors of A and B
* It is defined by the following formula

$similarity = cos(\theta) = \frac{A.B}{||A|| ||B||} = $
$\frac{\sum_{i=1}^nA_{i}B_{i}}
{\sqrt{\sum_{i=1}^nA_{i}^2}
\sqrt{\sum_{i=1}^nB_{i}^2}}$


* Closer the vectors, smaller will be the angle and larger the cosine

# **Pearson similarity**
* Similarity is the pearson coefficient between the two vectors.
* It is defined by the following formula

$r = $
$\frac{\sum_{i=1}^n(x_{i}-\overline{x})(y_{i}-\overline{y})}{{\sqrt{\sum_{i=1}^n(x_{i}-\overline{x})^2\sum_{i=1}^n(y_{i}-\overline{y})^2}}}$

# **Model Evaluation**
For evaluating recommendation engines, we can use the concept of precision-recall.

* **RMSE (Root Mean Squared Errors)**
  * Measures the error of predicted values
  * Lesser the RMSE value, better the recommendations

* **Recall**
  * What percentage of products that a user buys are actually recommended?
  * If a customer buys 5 products and the recommendation decided to show 3 of them, then the recall is 0.6

* **Precision**

  * Out of all the recommended items, how many the user actually liked?
  * If 5 products were recommended to the customer out of which he buys 4 of them, then precision is 0.8

* **Why are both recall and precision important?**

  * Consider a case where we recommend all products, so our customers will surely cover the items that they liked and bought. In this case, we have 100% recall! Does this mean our model is good?
  * We have to consider precision. If we recommend 300 items but user likes and buys only 3 of them, then precision is 0.1%! This very low precision indicates that the model is not great, despite their excellent recall.
  * So our aim has to be optimizing both recall and precision (to be close to 1 as possible).

Then we compare all the models we have built based on precision-recall characteristics, please refer attached Peapod_Recommendation_System.ipynb notebook for details.

# **Notes**
* **Popularity vs. Collaborative Filtering:** We can see that the collaborative filtering algorithms work better than popularity model for purchase counts. Indeed, popularity model doesn’t give any personalizations as it only gives the same list of recommended items to every user.

* **Precision and recall:** Looking at the summary above, we see that the precision and recall for Purchase Dummy > Normalized Purchase Counts.

* **RMSE:** Since RMSE is higher using pearson distance than cosine, we would choose model the smaller mean squared errors, which in this case would be cosine. Therefore, we select the Cosine similarity on Purchase Dummy approach as our final model.

# **Final Output**
* In this step, we would like to manipulate format for recommendation output to one we can export to output_option1_recommendation.csv 

* We need to first rerun the model using the whole dataset, as we came to a final model using train data and evaluated with test set.

# **Summary**

* In this exercise, we were able to traverse a step-by-step process for making recommendations to customers. 
* We used Collaborative Filtering approaches with cosine and pearson measure and compare the models with our baseline popularity model.
* We also prepared three sets of data that include regular buying count, buying dummy, as well as normalized purchase frequency as our target variable.
* Using RMSE, precision and recall, we evaluated our models and observed the impact of personalization. 
* Finally, we selected the Cosine approach in dummy purchase data.





