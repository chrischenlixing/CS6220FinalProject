# User Conversion API

This project provides a containerized Flask-based API for predicting TMall's user conversion using a pre-trained Random Forest model. The API processes JSON input and returns a prediction along with the confidence level.

---

### Prediction Endpoint from GPC

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

**Expected Output 2**:  
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

