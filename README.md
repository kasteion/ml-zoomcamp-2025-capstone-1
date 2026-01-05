# Sentiment Classification of User Posts

User-generated content such as posts, comments, and short messages contains valuable information about user opinions and experiences. Manually analyzing the sentiment of this text is time-consuming, subjective, and difficult to scale, especially as the volume of data grows.

This project addresses this problem by implementing a multi-class sentiment classification system that automatically categorizes user posts into negative, neutral, or positive sentiment. I tried different approaches to train the model:

- Logistic Regression counting words in the dataset
- Logistic Regression using tf-idf
- Decission tree classifier using word count and tf-idf
- Random forest classifier using word count and tf-idf
- XGBoost using word count and tf-idf
- Neural network using word count and tf-idf

Reading about this and based on the size of the dataset I learned that the best aproach for this task was to fine-tune a pretrained transformer like DistilBERT. So the final model is based on that because DistlBERT:

- Is small and fast
- Pretrained on masive text
- Needs very litle data
- Great for sentiment analysis

The machine learning workflow includes:

- Exploratory Data Analysis (EDA) to understand the dataset, visualize data distribution
- Data cleaning to encode the sentiment labels and preprocess the text (lowercase, remove puctuation marks, remove stop words, stem words)
- Training multiple ML models including Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network, and Fine-Tune a Pretrained transformer model.
- Selecting the fine-tuned DistilBERT as the best-performing model, based on f1_macro metric.
- Deploying the final model using FastAPI, making predictions accessible via an HTTP endpoint

## Dataset

For this project I used the following dataset: [data/sentiment_analysis.csv](https://github.com/kasteion/ml-zoomcamp-2025-capstone-1/blob/main/data/sentiment_analysis.csv)

## Installation and Setup

This project uses uv for dependency management, virtual environments, and project execution.

If you don’t have uv installed yet, follow the official instructions:

- https://github.com/astral-sh/uv

1. Clone the repo

```bash
git clone https://github.com/kasteion/ml-zoomcamp-2025-capstone-1.git
cd ml-zoomcamp-2025-capstone-1
```

2. Install dependencies

```bash
uv sync --locked
```

3. Run the train script

```bash
uv run train.py
```

4. Run the prediction service

```bash
uv run predict.py
```

## Docker

1. Build the Docker image

```bash
docker build -t sentiment-classifier .
```

2. Run the Docker container

```bash
docker run -it --rm -p 9696:9696 sentiment-classifier
```

## URL for testing

https://ml-zoomcamp-2025-capstone-1.fly.dev/predict

```bash
curl -X 'POST' \
  'https://ml-zoomcamp-2025-capstone-1.fly.dev/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Hello World!"
  }'
```

Or try it out through the swagger docs

https://ml-zoomcamp-2025-capstone-1.fly.dev/docs#/default/predict_predict_post

![swagger docs](https://github.com/kasteion/ml-zoomcamp-2025-capstone-1/blob/main/assets/swagger_docs.png)
