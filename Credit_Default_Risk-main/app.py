import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── Load Model & Scaler ───────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(page_title="Credit Default Predictor", page_icon="💳", layout="wide")
st.title("💳 Credit Card Default Predictor")
st.markdown("Fill in the customer details to predict default risk for next month.")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Info")
    LIMIT_BAL = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=50000, step=10000)
    SEX       = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    EDUCATION = st.selectbox("Education", [1, 2, 3, 4],
                             format_func=lambda x: {1:"Graduate School", 2:"University", 3:"High School", 4:"Others"}[x])
    MARRIAGE  = st.selectbox("Marital Status", [1, 2, 3],
                             format_func=lambda x: {1:"Married", 2:"Single", 3:"Others"}[x])
    AGE       = st.slider("Age", 18, 80, 30)

with col2:
    st.subheader("📅 Repayment Status")
    st.caption("-2=No consumption, -1=Paid in full, 0=Min paid, 1–9=Months delayed")
    PAY_0 = st.slider("Sept 2005 (PAY_0)", -2, 9, 0)
    PAY_2 = st.slider("Aug 2005  (PAY_2)", -2, 9, 0)
    PAY_3 = st.slider("Jul 2005  (PAY_3)", -2, 9, 0)
    PAY_4 = st.slider("Jun 2005  (PAY_4)", -2, 9, 0)
    PAY_5 = st.slider("May 2005  (PAY_5)", -2, 9, 0)
    PAY_6 = st.slider("Apr 2005  (PAY_6)", -2, 9, 0)

with col3:
    st.subheader("💰 Bill Amounts (NT$)")
    BILL_AMT1 = st.number_input("Bill Sept", 0, 1000000, 10000, step=1000)
    BILL_AMT2 = st.number_input("Bill Aug",  0, 1000000, 10000, step=1000)
    BILL_AMT3 = st.number_input("Bill Jul",  0, 1000000, 10000, step=1000)
    BILL_AMT4 = st.number_input("Bill Jun",  0, 1000000, 10000, step=1000)
    BILL_AMT5 = st.number_input("Bill May",  0, 1000000, 10000, step=1000)
    BILL_AMT6 = st.number_input("Bill Apr",  0, 1000000, 10000, step=1000)

st.markdown("---")
col4, col5 = st.columns(2)

with col4:
    st.subheader("💸 Previous Payments (NT$)")
    PAY_AMT1 = st.number_input("Paid Sept", 0, 1000000, 2000, step=500)
    PAY_AMT2 = st.number_input("Paid Aug",  0, 1000000, 2000, step=500)
    PAY_AMT3 = st.number_input("Paid Jul",  0, 1000000, 2000, step=500)

with col5:
    st.subheader(" ")
    PAY_AMT4 = st.number_input("Paid Jun", 0, 1000000, 2000, step=500)
    PAY_AMT5 = st.number_input("Paid May", 0, 1000000, 2000, step=500)
    PAY_AMT6 = st.number_input("Paid Apr", 0, 1000000, 2000, step=500)

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────────
if st.button("🔍 Predict Default Risk", use_container_width=True, type="primary"):

    bill_cols  = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
    pay_cols   = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    delay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

    input_data = {
        "LIMIT_BAL": LIMIT_BAL, "SEX": SEX, "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,   "AGE": AGE,
        "PAY_0": PAY_0, "PAY_2": PAY_2, "PAY_3": PAY_3,
        "PAY_4": PAY_4, "PAY_5": PAY_5, "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2, "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4, "BILL_AMT5": BILL_AMT5, "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1,   "PAY_AMT2": PAY_AMT2,   "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4,   "PAY_AMT5": PAY_AMT5,   "PAY_AMT6": PAY_AMT6,
    }

    df = pd.DataFrame([input_data])

    # Feature engineering — must match notebook exactly
    df["AVG_BILL"]       = df[bill_cols].mean(axis=1)
    df["AVG_PAYMENT"]    = df[pay_cols].mean(axis=1)
    df["PAY_BILL_RATIO"] = df["AVG_PAYMENT"] / (df["AVG_BILL"].abs() + 1)
    df["DELAY_SCORE"]    = df[delay_cols].sum(axis=1)

    for col in ["LIMIT_BAL", "AVG_BILL", "AVG_PAYMENT"] + bill_cols + pay_cols:
        df[col] = np.log1p(df[col].clip(lower=0))

    scaled = scaler.transform(df)
    prob   = model.predict_proba(scaled)[0][1]
    pred   = int(prob >= 0.68)

    # ── Results ───────────────────────────────────────────────────
    st.markdown("## 🧾 Result")
    r1, r2, r3 = st.columns(3)

    with r1:
        if pred == 1:
            st.error("⚠️ HIGH RISK — Likely to Default")
        else:
            st.success("✅ LOW RISK — Unlikely to Default")
    with r2:
        st.metric("Default Probability", f"{prob*100:.1f}%")
    with r3:
        st.metric("Threshold Used", "68%")

    st.markdown("#### Risk Probability")
    st.progress(float(prob))

    with st.expander("📊 Feature Summary"):
        avg_bill    = np.mean([BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6])
        avg_pay     = np.mean([PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6])
        delay_score = PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6
        util        = BILL_AMT1 / (LIMIT_BAL + 1) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Bill",           f"NT${avg_bill:,.0f}")
        c2.metric("Avg Payment",        f"NT${avg_pay:,.0f}")
        c3.metric("Delay Score",        f"{delay_score}")
        c4.metric("Credit Utilization", f"{util:.1f}%")

        if delay_score > 3:
            st.warning("🔴 High delay score — multiple months of late payments.")
        if util > 80:
            st.warning("🔴 High credit utilization — bill close to credit limit.")
        if avg_pay < avg_bill * 0.1:
            st.warning("🔴 Low payment-to-bill ratio — barely paying minimum.")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Predicts credit card default using **Gradient Boosting** trained on the UCI Credit Card dataset (30,000 records).")
    st.markdown("---")
    st.markdown("**Model:** GradientBoostingClassifier")
    st.markdown("**Scaler:** RobustScaler")
    st.markdown("**Threshold:** 0.68")
