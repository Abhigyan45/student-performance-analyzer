# 📊 Automated Student Performance Analyzer

A machine learning web application that analyzes student academic data, generates visual performance reports, and identifies at-risk students using **K-Means Clustering**.

## 🚀 Live Demo
> Deploy on [Streamlit Cloud](https://streamlit.io/cloud) for free

## 📌 Features
- Upload any student CSV file (marks + attendance)
- Auto-generates subject-wise performance reports
- K-Means clustering to group students by performance
- Flags at-risk students based on customizable threshold
- Downloadable full report as CSV
- Interactive charts: bar, box, heatmap, scatter, pie

## 🛠️ Tech Stack
- **Python** — core language
- **Streamlit** — web UI
- **Pandas & NumPy** — data processing
- **Scikit-learn** — K-Means clustering
- **Matplotlib & Seaborn** — visualizations

## 📂 CSV Format
Your CSV should have this structure:
```
Name, Math, Science, English, History, Attendance
Alice, 85, 90, 78, 88, 95
Bob, 42, 38, 45, 40, 60
```
Subjects can be any name — the app auto-detects them.

## ⚙️ Installation & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/student-performance-analyzer.git
cd student-performance-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Click Deploy — done!

## 📸 Screenshots
> Add screenshots of your app here after running it

## 👨‍💻 Author
**Ranjeet Abhigyan**  
M.Tech – AI & Data Science, IIIT Bhagalpur  
[LinkedIn](#) | [GitHub](#)
