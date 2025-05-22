import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="🎓 Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance Analysis")
st.markdown("An interactive dashboard to explore and analyze student academic performance.")

# 📂 File uploader
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# 📁 Sample dataset
@st.cache_data
def load_sample_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

# 🧹 Prepare data
def clean_data(df):
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    if "study_hours" not in df.columns:
        np.random.seed(42)
        df["study_hours"] = np.random.normal(loc=5, scale=2, size=len(df)).round(1)
        df["study_hours"] = df["study_hours"].clip(lower=0)
    return df

# 🧠 Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully")
else:
    st.sidebar.info("ℹ️ Using sample dataset")
    df = load_sample_data()

df = clean_data(df)

# ℹ️ Dataset info
st.subheader("📑 Dataset Preview")
st.dataframe(df.head())

# 📌 Filters
st.sidebar.header("Filter Data")
gender_filter = st.sidebar.multiselect("Select Gender", df["gender"].unique(), default=df["gender"].unique())
race_filter = st.sidebar.multiselect("Select Race/Ethnicity", df["race/ethnicity"].unique(), default=df["race/ethnicity"].unique())
edu_filter = st.sidebar.multiselect("Parental Education", df["parental_level_of_education"].unique(), default=df["parental_level_of_education"].unique())

filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["race/ethnicity"].isin(race_filter)) &
    (df["parental_level_of_education"].isin(edu_filter))
]

# 📊 Tabs for layout
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visual Analysis", "📉 Regression", "📊 Distribution", "🚨 At-Risk Students"])

# 📈 Tab 1: Correlation Heatmap
with tab1:
    st.subheader("📈 Subject Score Correlation Heatmap")
    numeric_df = filtered_df[["math_score", "reading_score", "writing_score"]]
    fig1, ax1 = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

# 📉 Tab 2: Regression - Study Hours vs Math Score
with tab2:
    st.subheader("📉 Regression: Study Hours vs Math Score")
    X = filtered_df[["study_hours"]]
    y = filtered_df["math_score"]
    model = LinearRegression().fit(X, y)
    predicted = model.predict(X)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="study_hours", y="math_score", data=filtered_df, ax=ax2, label="Actual")
    sns.lineplot(x=filtered_df["study_hours"], y=predicted, color="red", ax=ax2, label="Predicted")
    ax2.set_title(f"Regression Line (R²: {model.score(X, y):.2f})")
    st.pyplot(fig2)

# 📊 Tab 3: Score Distribution
with tab3:
    st.subheader("📊 Score Distribution")
    score_choice = st.selectbox("Choose score to visualize", ["math_score", "reading_score", "writing_score"])
    fig3, ax3 = plt.subplots()
    sns.histplot(filtered_df[score_choice], kde=True, ax=ax3, bins=20)
    ax3.set_title(f"{score_choice.replace('_', ' ').title()} Distribution")
    st.pyplot(fig3)

# 🚨 Tab 4: At-Risk Students
with tab4:
    st.subheader("🚨 At-Risk Students (Math Score < 40)")
    at_risk = filtered_df[filtered_df["math_score"] < 40]
    st.warning(f"Found {len(at_risk)} students with Math Score below 40.")
    st.dataframe(at_risk)

# 📥 Download filtered data
st.sidebar.markdown("### 📥 Download Filtered Data")
st.sidebar.download_button("Download CSV", filtered_df.to_csv(index=False), "filtered_data.csv", "text/csv")

# 📘 Dataset format info
st.sidebar.markdown("""
---
### 📘 Dataset Format  
Your CSV should include:
- `gender`  
- `race/ethnicity`  
- `parental level of education`  
- `math score`, `reading score`, `writing score`  
Optional: `study_hours`
""")


