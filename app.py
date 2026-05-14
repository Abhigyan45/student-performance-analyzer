import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(page_title="Student Performance Analyzer", page_icon="📊", layout="wide")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1F4E79; text-align: center; }
    .subtitle   { font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .risk-high  { background-color: #FFDADA; padding: 8px 14px; border-radius: 8px; color: #c0392b; font-weight: bold; }
    .risk-low   { background-color: #D4EFDF; padding: 8px 14px; border-radius: 8px; color: #1e8449; font-weight: bold; }
    .metric-box { background: #EBF5FB; border-radius: 10px; padding: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 Automated Student Performance Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload student data to generate visual reports and identify at-risk students using K-Means Clustering</p>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 5, 2)
risk_threshold = st.sidebar.slider("At-Risk Threshold (avg marks < )", 30, 60, 40)
st.sidebar.markdown("---")
st.sidebar.markdown("**📥 Sample CSV Format:**")
st.sidebar.code("Name, Math, Science, English, History, Attendance")

# ─── Sample Data Generator ────────────────────────────────────────────────────
def generate_sample_data():
    np.random.seed(42)
    names = [f"Student_{i+1}" for i in range(50)]
    data = {
        "Name": names,
        "Math":       np.random.randint(20, 100, 50),
        "Science":    np.random.randint(20, 100, 50),
        "English":    np.random.randint(20, 100, 50),
        "History":    np.random.randint(20, 100, 50),
        "Attendance": np.random.randint(50, 100, 50),
    }
    return pd.DataFrame(data)

# ─── File Upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload Student CSV File", type=["csv"])

if uploaded_file is None:
    st.info("No file uploaded. Using sample dataset (50 students).")
    df = generate_sample_data()
else:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

st.success(f"✅ Loaded {len(df)} student records successfully!")

# ─── Data Preview ─────────────────────────────────────────────────────────────
with st.expander("📋 Preview Raw Data"):
    st.dataframe(df, use_container_width=True)

# ─── Feature Engineering ──────────────────────────────────────────────────────
subject_cols = [c for c in df.columns if c not in ["Name", "Attendance"]]
df["Average_Marks"] = df[subject_cols].mean(axis=1).round(2)
df["At_Risk"] = df["Average_Marks"].apply(lambda x: "⚠️ At Risk" if x < risk_threshold else "✅ Safe")

# ─── Summary Metrics ──────────────────────────────────────────────────────────
st.markdown("### 📈 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students",   len(df))
col2.metric("Average Score",    f"{df['Average_Marks'].mean():.1f}")
col3.metric("At-Risk Students", len(df[df["At_Risk"] == "⚠️ At Risk"]))
col4.metric("Top Scorer",       f"{df['Average_Marks'].max():.1f}")

st.markdown("---")

# ─── K-Means Clustering ───────────────────────────────────────────────────────
features = df[subject_cols + ["Attendance"]].fillna(0)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Label clusters by average marks
cluster_avg = df.groupby("Cluster")["Average_Marks"].mean()
sorted_clusters = cluster_avg.sort_values().index.tolist()
label_map = {}
for i, c in enumerate(sorted_clusters):
    if i == 0:
        label_map[c] = "🔴 Low Performer"
    elif i == len(sorted_clusters) - 1:
        label_map[c] = "🟢 High Performer"
    else:
        label_map[c] = f"🟡 Average (Group {i})"
df["Performance_Group"] = df["Cluster"].map(label_map)

# ─── Visualizations ───────────────────────────────────────────────────────────
st.markdown("### 📊 Visual Reports")

tab1, tab2, tab3, tab4 = st.tabs(["📉 Subject Trends", "🎯 Clustering", "⚠️ At-Risk Students", "📋 Full Report"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Marks by Subject**")
        fig, ax = plt.subplots(figsize=(6, 4))
        subject_means = df[subject_cols].mean()
        bars = ax.bar(subject_means.index, subject_means.values,
                      color=sns.color_palette("Blues_d", len(subject_cols)))
        ax.set_ylabel("Average Marks")
        ax.set_title("Subject-wise Performance")
        for bar, val in zip(bars, subject_means.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Marks Distribution (Box Plot)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        df[subject_cols].boxplot(ax=ax, patch_artist=True)
        ax.set_title("Score Distribution by Subject")
        ax.set_ylabel("Marks")
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df[subject_cols + ["Attendance", "Average_Marks"]].corr(),
                annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Student Clusters (Avg Marks vs Attendance)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
        for i, (grp, gdf) in enumerate(df.groupby("Performance_Group")):
            ax.scatter(gdf["Attendance"], gdf["Average_Marks"],
                       label=grp, color=colors[i % len(colors)], alpha=0.8, s=60)
        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Average Marks")
        ax.set_title("K-Means Clustering Result")
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Performance Group Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        grp_counts = df["Performance_Group"].value_counts()
        ax.pie(grp_counts.values, labels=grp_counts.index,
               autopct="%1.1f%%", colors=colors[:len(grp_counts)],
               startangle=90)
        ax.set_title("Student Group Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab3:
    at_risk_df = df[df["At_Risk"] == "⚠️ At Risk"][["Name"] + subject_cols + ["Average_Marks", "Attendance", "Performance_Group"]]
    st.markdown(f"### ⚠️ {len(at_risk_df)} Students Identified as At-Risk (Avg < {risk_threshold})")
    st.dataframe(at_risk_df.sort_values("Average_Marks").reset_index(drop=True), use_container_width=True)

    if len(at_risk_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(at_risk_df["Name"], at_risk_df["Average_Marks"], color="#e74c3c", alpha=0.8)
        ax.axvline(x=risk_threshold, color="black", linestyle="--", label=f"Threshold ({risk_threshold})")
        ax.set_xlabel("Average Marks")
        ax.set_title("At-Risk Students — Average Marks")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab4:
    st.markdown("**Complete Student Report**")
    report_df = df[["Name"] + subject_cols + ["Average_Marks", "Attendance", "At_Risk", "Performance_Group"]]
    st.dataframe(report_df.sort_values("Average_Marks").reset_index(drop=True), use_container_width=True)

    # Download CSV
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Full Report as CSV",
        data=csv_buffer.getvalue(),
        file_name="student_performance_report.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("<center>Built with ❤️ using Python, Streamlit, Scikit-learn | Ranjeet Abhigyan</center>", unsafe_allow_html=True)
