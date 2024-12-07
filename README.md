# User Conversion API

This project provides a containerized Flask-based API for predicting TMall's user conversion using a pre-trained Random Forest model. The API processes JSON input and returns a prediction along with the confidence level.

---

### Team Members
Lingling Deng, Chih-Yun Chang, Yao Chen, Lixing Chen

### Link To Final Presentation

[Presentation Link](https://docs.google.com/presentation/d/1cb-_aNdjuvIsOtbXhIDQndhNEeoCvosx/edit#slide=id.p1)

### Link To Final Report

[Report Link](https://docs.google.com/document/d/1Ts4IfIu7GuhiY29eZIie_85A4pEZpQRS0t2uK3otXqI/edit?tab=t.0)

---

### Prediction Endpoint from GCP

**URL**: Open your browser or API testing tool (e.g., Postman) and send a `POST` request to `https://user-conversion-api-999732714550.us-east1.run.app/predict`

**Method**: `POST`

**Headers**:  
```json
{
  "Content-Type": "application/json"
}
```

**Sample Input 1**:  
```json
{
    "Income": 39192,
    "CampaignChannel": 0,
    "CampaignType": 0,
    "AdSpend": 429.07921736744015,
    "ClickThroughRate": 0.0728897285108404,
    "ConversionRate": 0.09538896175545068,
    "WebsiteVisits": 44,
    "PagesPerVisit": 6.95369491204816,
    "TimeOnSite": 2.2520393830350898,
    "SocialShares": 33,
    "EmailOpens": 0,
    "EmailClicks": 1,
    "PreviousPurchases": 1,
    "LoyaltyPoints": 4474
}
```

**Expected Output 1**:  
```json
{
    "confidence": 0.88,
    "prediction": 0
}
```
**Sample Input 2**:  
```json
{
    "Income": 55972.0,
    "AdSpend": 7254.022157320001,
    "ClickThroughRate": 0.083025001631625,
    "ConversionRate": 0.015726698375027437,
    "WebsiteVisits": 38,
    "PagesPerVisit": 5.447257523959778,
    "TimeOnSite": 6.134174721060454,
    "SocialShares": 79,
    "EmailOpens": 2,
    "EmailClicks": 8,
    "PreviousPurchases": 1,
    "LoyaltyPoints": 612
}
```
**Expected Output 2**:  
```json
{
    "confidence": 0.82,
    "prediction": 0
}
```

**Sample Input 3**:  
```json
{
    "Income": 109779.0,
    "AdSpend": 8383.984491524046,
    "ClickThroughRate": 0.28263938781591186,
    "ConversionRate": 0.17492572533137893,
    "WebsiteVisits": 6,
    "PagesPerVisit": 2.384030881810251,
    "TimeOnSite": 12.561754154488309,
    "SocialShares": 97,
    "EmailOpens": 14,
    "EmailClicks": 2,
    "PreviousPurchases": 3,
    "LoyaltyPoints": 467
}
```   

**Expected Output 3**:  
```json
{
    "confidence": 0.79,
    "prediction": 1
}
```

**Sample Input 4**:  
```json
{
  "Income": 136912,
  "AdSpend": 6497.870068417766,
  "ClickThroughRate": 0.04391851073538301,
  "ConversionRate": 0.08803141207288108,
  "WebsiteVisits": 0,
  "PagesPerVisit": 2.399016527783845,
  "TimeOnSite": 7.3968025807960585,
  "SocialShares": 19,
  "EmailOpens": 6,
  "EmailClicks": 9,
  "PreviousPurchases": 4,
  "LoyaltyPoints": 688
}
```

**Expected Output 4**:  
```json
{
    "confidence": 0.94,
    "prediction": 1
}
```

---

## Requirements

### To run locally:

- Python 3.9+
- Docker

---

## Setup

### Running Locally

1. **Clone the repository**:

  ```bash
  git clone https://github.com/chrischenlixing/CS6220FinalProject.git
  cd CS6220FinalProject
  ```

2. **Install dependencies**:

  ```bash
  pip install -r requirements.txt
  ```

3. **Generate the model file**:
  ```bash
  python user_conversion.py
  ```

4. **Run the Flask application**:
  ```bash
  python app.py
  ```

5. **Access the API:**:
  Open your browser or API testing tool (e.g., Postman) and send a `POST` request to `http://127.0.0.1:8080/predict`

### Running with Docker

1. **Build the Docker image:**:
  
  ```bash
  docker build -t user-conversion-api .
  ```

2. **Run the Docker container:**:
  
  ```bash
  docker run -p 8080:8080 user-conversion-api
  ```
---

