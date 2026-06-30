# 💳 Credit Card Default Predictor

A machine learning web application that predicts whether a credit card customer will default on their payment next month, built with **Gradient Boosting** and deployed using **Streamlit**.

---

## 📁 Project Structure

```
project/
├── app.py                  # Streamlit web application
├── model.pkl               # Trained GradientBoostingClassifier
├── scaler.pkl              # Fitted RobustScaler
├── Credit_default.ipynb    # Model training notebook
├── UCI_Credit_Card.csv     # Dataset
└── README.md
```

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Records:** 30,000 credit card customers
- **Target:** `default.payment.next.month` (1 = default, 0 = no default)
- **Class distribution:** ~78% No Default / ~22% Default (imbalanced)

---

## ⚙️ How It Works

### Preprocessing
- Cleaned outlier categories in `EDUCATION` (0, 5, 6 → 4) and `MARRIAGE` (0 → 3)
- Engineered new features: `AVG_BILL`, `AVG_PAYMENT`, `PAY_BILL_RATIO`, `DELAY_SCORE`
- Applied `log1p` transform on skewed columns (`LIMIT_BAL`, bill amounts, payment amounts)
- Scaled features using `RobustScaler`

### Model
- **Algorithm:** GradientBoostingClassifier (scikit-learn)
- **Class imbalance handling:** `compute_sample_weight(class_weight='balanced')`
- **Decision threshold:** 0.68 (tuned for optimal precision/recall balance)

### Metrics
| Metric | Score |
|---|---|
| Accuracy | ~82% |
| ROC-AUC | ~0.78 |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy
```

### 2. Place files in the same folder

```
app.py
model.pkl
scaler.pkl
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push your project folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → connect your GitHub repo
4. Set `app.py` as the main file → click **Deploy**

> ⚠️ Make sure `model.pkl` and `scaler.pkl` are committed to the repo.

---

## 🖥️ App Features

- Input form for all customer details — personal info, repayment history, bill and payment amounts
- Real-time default probability prediction
- Visual risk indicator (progress bar)
- Feature summary with automatic risk warnings:
  - High delay score (multiple late payments)
  - High credit utilization (bill close to limit)
  - Low payment-to-bill ratio

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
```

Install all at once:

```bash
pip install streamlit scikit-learn pandas numpy
```

---

## 📌 Notes

- The model was trained on data from **April–September 2005**
- Threshold of **0.68** was chosen to balance precision and recall for the default class
- For credit risk use cases, lowering the threshold (e.g. to 0.3–0.4) increases recall (catches more defaulters) at the cost of some accuracy
