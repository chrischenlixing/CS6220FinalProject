# User Conversion API

This project provides a containerized Flask-based API for predicting user conversion using a pre-trained Random Forest model. The API processes JSON input and returns a prediction along with the confidence level.

---

### Prediction Endpoint from GPC

**URL**: Open your browser or API testing tool (e.g., Postman) and send a `POST` request to `https://user-conversion-api-999732714550.us-east1.run.app/predict` 
**Method**: `POST`  
**Headers**:  
```json
{
  "Content-Type": "application/json"
}
**Sample Input**:  
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
  git clone https://github.com/your-username/your-repo.git
  cd your-repo

2. **Install dependencies**:

  ```bash
  pip install -r requirements.txt
  

3. **Generate the model file**:
  ```bash
  python user_conversion.py

4. **Run the Flask application**:
  ```bash
  python app.py

5. **Access the API:**:
  Open your browser or API testing tool (e.g., Postman) and send a `POST` request to `http://127.0.0.1:8080/predict`

### Running with Docker

1. **Build the Docker image:**:
  
  ```bash
  docker build -t user-conversion-api .


2. **Run the Docker container:**:
  
  ```bash
  docker run -p 8080:8080 user-conversion-api

---

