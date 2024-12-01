# User Conversion API

This project provides a containerized Flask-based API for predicting user conversion using a pre-trained Random Forest model. The API processes JSON input and returns a prediction along with the confidence level.

---

## Features

- **Prediction Endpoint**: `/predict` - Accepts JSON input and returns a prediction.
- **Containerized Deployment**: Runs on Google Cloud Run for scalable and managed hosting.
- **Logging**: Uses Cloud Logging for robust monitoring.

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

2. **Install dependencies:**:

   ```bash
   pip install -r requirements.txt
  

3. **Generate the model file:**:
  ```bash
  python user_conversion.py