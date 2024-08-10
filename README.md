# Thompson-Okoka-ML-DS-Portfolio
![ML_Pix](https://github.com/user-attachments/assets/d5acfff3-2621-4d01-8ced-a002542218e4)

This repository is to act like a high level overview of the data science college projects I've worked on with my team: Tavleen and Sachin.

Personal Statement: Thompson Okoka is a passionate Machine Learning and Data Science professional with a strong foundation in applied machine learning, data analytics, and business problem-solving. My portfolio showcases a diverse range of machine learning and data science projects that demonstrate my ability to apply advanced analytics techniques to real-world problems across different domains, from social media analytics to customer churn prediction and cybersecurity.

**Contact Information:**

**Email:** thompsonokoka@gmail.com

**LinkedIn:** https://www.linkedin.com/in/thompsonokoka/

**GitHub:** https://github.com/Tokoka2024

**Website:** https://github.com/Tokoka2024


**Skills:**

* Programming Languages: Python, SQL
* Machine Learning Frameworks: scikit-learn, TensorFlow, Keras
* Data Visualization: Matplotlib, Seaborn, Plotly
* Tools: Jupyter Notebook, Git, Docker, Pandas, NumPy
* Database Management: MySQL, PostgreSQL
* Deployment: Flask, Heroku

**Career Goals:**

I am looking forward to leveraging my expertise in machine learning and data science to solve complex business problems and drive meaningful insights. My aim is to work in a challenging environment where I can contribute to innovative projects and grow as a professional in this ever-evolving field.

**Call to Action:**

I am always open to discussing new opportunities and collaborations. If you are looking for a data-driven professional with a proven track record in applying machine learning to real-world challenges, feel free to reach out to me.

**Projects Overview:**

The portfolio includes six(6) distinct projects covering different areas:

1. User engagement prediction (social media analytics)
2. Customer churn prediction (e-commerce)
3. Cybersecurity anomaly detection (healthcare)
4. Plant disease detection (agriculture)
5. Appointment no-show prediction for Hairsalon (service industry)
6. Customer engagement prediction (Lifestyle/Commerce)

This variety demonstrates versatility and the ability to apply ML techniques to different sectors.
![image](https://github.com/user-attachments/assets/96c8d5d4-2695-4718-b633-b21b6432e4bf)

# Project 1: User Engagement Prediction for TikTok

**Description:**
This project involved predicting user engagement on TikTok by analyzing various features such as average engagement rate, number of likes, comments, and followers. The goal was to determine the most effective model for predicting engagement rates, which could inform business decisions regarding content personalization and ad targeting.

**Dataset:**

•	**Key Features:**  Average engagement rate, comment engagement rate, like engagement rate, number of followers, number of accounts following, and total likes.
![image](https://github.com/user-attachments/assets/4945c49c-f315-4137-84eb-a2194b059bcc)

•	**Size:**  Comprehensive dataset with key engagement metrics for various TikTok accounts.

![image](https://github.com/user-attachments/assets/aa815821-226c-48b5-8574-373bc35f537a)

![Tiktok_Influencer](https://github.com/user-attachments/assets/87568bcd-a3b0-4dd0-a40e-6871684b536b)

<img width="471" alt="image" src="https://github.com/user-attachments/assets/c0432f59-b02f-46e8-8211-c1b64ed09d84">


**Methodology:**

•	**Models Used:**
1. Linear Regression

    <img width="461" alt="image" src="https://github.com/user-attachments/assets/c833612e-0f7b-4916-b9c2-37aa1ca77e41">

2. RandomForestRegressor

   <img width="466" alt="image" src="https://github.com/user-attachments/assets/8b0c427a-8438-405c-9789-7b38973fd126">

3. GradientBoostingRegressor

<img width="568" alt="image" src="https://github.com/user-attachments/assets/4b7d961e-96f3-4db1-80c7-cfd02fb15f3b">

   
•	**Evaluation Metrics:**

1. Mean Squared Error (MSE)
2. Root Mean Squared Error (RMSE)
3. R-squared (R2)
4. Cross-Validation R2 Mean (CV_R2_mean)
<img width="626" alt="image" src="https://github.com/user-attachments/assets/228fee7b-42a5-44ee-a486-d84a3a4af0c1">

 **Outcome:**

•	**Best Model:** GradientBoostingRegressor outperformed other models with the lowest MSE and RMSE, and the highest R2.

•	**Business Recommendations:**

1. Content Personalization: Use Gradient Boosting for tailored content recommendations to boost user engagement.
2. Influencer Support: Identify and support top influencers based on model predictions to maintain high engagement levels.
3. Ad Targeting: Enhance ad targeting strategies by predicting user preferences and behavior, leading to higher revenue.
   
Link to Project: https://github.com/Tokoka2024/CSCN8030---Spring-2024---Section-2.

![image](https://github.com/user-attachments/assets/29b2de4c-d036-45dd-9b62-f56c274866e2)



# Project 2: Customer Churn Prediction for Loblaw Digital
![image](https://github.com/user-attachments/assets/e986becb-610a-4e3a-ae67-81dc449f065f)


**Description:** This project aimed to predict customer churn for Loblaw Digital, the e-commerce arm of one of Canada's largest retail companies. By analyzing customer data, the goal was to develop a model that could predict churn and help the company implement proactive retention strategies.

**Dataset:**

* **Key Features:** Customer tenure, preferred login device, city tier, payment mode, satisfaction score, complaints, order count, cashback amount, etc.
* * **Size:** Detailed customer data with a focus on identifying churn predictors.

   <img width="532" alt="image" src="https://github.com/user-attachments/assets/a4476620-06e3-4c82-b5ba-814c16158e61">

<img width="560" alt="image" src="https://github.com/user-attachments/assets/bae554a7-5473-49d8-a183-c854659dc0f6">


**Methodology:**

*	**Models Used:**
    1. Random Forest Classifier
       <img width="557" alt="image" src="https://github.com/user-attachments/assets/e1887f11-403a-41b1-bc63-27a6814c47b0">

    2. Gradient Boosting Classifier
       <img width="560" alt="image" src="https://github.com/user-attachments/assets/436fa88e-de96-4e0c-b931-5e85880f34af">

   
*	**Evaluation Metrics:**

    * Accuracy
    * Precision
    * Recall
    * F1 Score

**Outcome:**

* **Best Model:** Random Forest Classifier achieved the highest accuracy of 96%, making it the preferred model for churn prediction.

* **Personalized Retention Strategy:**
  
    * **Tiered Loyalty Program:** Implement a tiered program based on churn risk and customer lifetime value (CLV).
    * **Personalized Product Recommendations:** Use collaborative filtering to recommend products based on user preferences and past behavior.

Link to Project: https://github.com/Tokoka2024/CSCN8030---Spring-2024---Section-2

# ![image](https://github.com/user-attachments/assets/e8ad5432-df99-452b-a392-99da5f439363)

# Project 3: Cybersecurity in Small Businesses - Maven Clinic
**Description:** In this project, machine learning techniques were applied to enhance cybersecurity for Maven Clinic, a healthcare startup providing telemedicine services. The focus was on anomaly detection to identify potential security threats and ensure data privacy.

**Dataset:**

* **Key Features:** Provider details, service codes, payment amounts, etc.
* **Size:** Healthcare provider data with focus on security.

<img width="607" alt="image" src="https://github.com/user-attachments/assets/8e59ecd7-a72f-4e0e-8501-9c8f145983cf">

  <img width="599" alt="image" src="https://github.com/user-attachments/assets/948aaf59-d558-48cf-b20d-153758b7cec5">


**Methodology:**

* **Techniques Used:**
1. Fernet Encryption Algorithm for data protection.
2. Isolation Forest and DBSCAN for anomaly detection.
   
* **Evaluation Metrics:**
1. Anomaly detection accuracy
2. False positive rate
<img width="472" alt="image" src="https://github.com/user-attachments/assets/4b4c0b4a-3522-4772-88e2-5f130dab85eb">
        
**Outcome:**

 * **Security Enhancement:** The models successfully identified anomalies, helping to prevent potential security breaches.
 * **Use Cases:** The techniques were particularly effective in fraud detection, network security, and quality control.

Link to Project: https://github.com/Tokoka2024/CSCN8030---Spring-2024---Section-2

# ![image](https://github.com/user-attachments/assets/e8ad5432-df99-452b-a392-99da5f439363)
# Project 4: Plant Disease Detection for Small Businesses<img width="469" alt="image" src="https://github.com/user-attachments/assets/7df88191-d66d-4fd5-bc63-aa8497f3d1c0">

**Description:** This project focused on developing a machine learning solution for detecting plant diseases, aimed at helping small-scale farmers. The solution was designed to be cost-effective and accessible, providing early detection to improve crop yield and quality.

**Dataset:**

* Key Features: Images of plants labeled as 'Healthy', 'Powdery', or 'Rust'.
* Size: 1,530 plant images divided into training, testing, and validation sets.
<img width="515" alt="image" src="https://github.com/user-attachments/assets/f94cc770-1f37-48f9-95b2-c2aa7138422a">
<img width="665" alt="image" src="https://github.com/user-attachments/assets/61b6c4b4-c517-49f9-85f4-1eeb80d31813">

**Methodology:**

* **Models Used:**
  
    1. Custom Convolutional Neural Network (CNN)
    2. VGG16 (Transfer Learning)
   
* **Evaluation Metrics:**
    1. Accuracy
    2. Loss
<img width="746" alt="image" src="https://github.com/user-attachments/assets/0fc42ecd-8f28-45b1-87c6-bf9ea42b045a">
<img width="719" alt="image" src="https://github.com/user-attachments/assets/8684b9ec-3c0e-4675-a843-1d2b80a07840">
**Outcome:**

   * **Best Model:** The Custom CNN model outperformed the VGG16 model with a testing accuracy of 86% and a loss of 0.3286.
   * **Impact:** The solution provides small-scale farmers with a reliable tool for early disease detection, potentially saving crops and improving food security.

Link to Project: https://github.com/Tokoka2024/CSCN8030---Spring-2024---Section-2 

# ![image](https://github.com/user-attachments/assets/e8ad5432-df99-452b-a392-99da5f439363)

# Project 5: No-Show Prediction for TTSaloon
**Description:** This project aimed to predict appointment no-shows at a specialized hair salon, TTSaloon, using historical booking data. The objective was to reduce the occurrence of no-shows, which negatively impact business operations and customer satisfaction.

**Dataset:**

* **Key Features:** Booking time, day of the week, service category, staff assigned, last service details, cumulative revenue, etc.
* **Size:** Dataset covering bookings and cancellations from March to July 2018.

**Methodology:**

* **Models Used:**
    1. Logistic Regression
    2. K-Nearest Neighbors (KNN)
    3. Random Forest
    4. AdaBoost
    
* **Evaluation Metrics:**
    1. Accuracy
    2. Precision
    3. Recall
    4. ROC Curve

**Outcome:**

   * **Best Model:** Gradient Boosting exhibited the most balanced performance with an accuracy of 92%.
   * **Findings:** Gradient Boosting was most effective at handling class imbalance, making it the best model for predicting no-shows. Suggestions for further improvement                     include exploring resampling techniques.

Link to Project: https://github.com/Tokoka2024/CSCN8030---Spring-2024---Section-2

