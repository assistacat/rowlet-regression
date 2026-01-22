# 🎯 AI-Driven Company Intelligence Prototype

**NUS Datathon 2026 | Champions Group Challenge**

A universal tool designed for lead generation, company segmentation, and competitive intelligence. Upload firmographic data → filter → cluster → explore peers → get AI-powered insights.

## ✨ Features

- **Smart Data Processing**: Automated cleaning and validation of firmographic data
- **Advanced Filtering**: Multi-dimensional filters for industry, geography, revenue, and employees
- **K-Means Clustering**: Intelligent company segmentation with visual PCA representation
- **Peer Benchmarking**: Compare companies against cluster averages with radar charts
- **AI-Powered Insights**: GPT-driven business intelligence analysis with actionable recommendations
- **Anomaly Detection**: Identify outliers and unusual patterns within clusters
- **Professional UI**: Dark-themed interface with interactive visualizations

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/assistacat/rowlet-regression.git
cd rowlet-regression
```

### 2. Install Dependencies

Ensure you have **Python 3.9+** installed, then install required packages:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit pandas numpy scikit-learn plotly openpyxl groq
```

### 3. Set Up Groq API Key

This app uses Groq AI for generating business insights. Follow these steps:

1. **Get a Groq API Key**:
   - Visit [https://console.groq.com](https://console.groq.com)
   - Sign up or log in to your account
   - Navigate to API Keys section
   - Create a new API key and copy it

2. **Create Secrets Configuration**:
   - In your project directory, create a `.streamlit` folder if it doesn't exist:
     ```bash
     mkdir .streamlit
     ```
   
   - Create a file named `secrets.toml` inside the `.streamlit` folder:
     ```bash
     # Windows
     type nul > .streamlit\secrets.toml
     
     # Mac/Linux
     touch .streamlit/secrets.toml
     ```

3. **Add Your API Key**:
   - Open `.streamlit/secrets.toml` in a text editor
   - Add the following line (replace with your actual key):
     ```toml
     GROQ_API_KEY = "gsk_your_actual_api_key_here"
     ```
   
   - Save the file

> **Note**: Never commit `secrets.toml` to version control. Add `.streamlit/secrets.toml` to your `.gitignore` file.

### 4. Run the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Usage

1. **Upload Data**: Use the sidebar to upload a CSV or Excel file with company data
2. **Apply Filters**: Filter by industry (NAICS), country, revenue range, and employee count
3. **Run Clustering**: Click "Apply K-Means Clustering" to segment companies
4. **Explore Insights**: 
   - View cluster profiles and benchmarking statistics
   - Select individual companies to see detailed comparisons
   - Generate AI-powered business insights (saved with timestamps)
   - Identify anomalies and outliers

## 📁 Project Structure

```
rowlet-regression/
├── app.py                  # Main Streamlit application
├── data_prep.py            # Data cleaning and preprocessing
├── model_day1.py           # Clustering and PCA visualization
├── insights.py             # Radar charts and anomaly detection
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── .streamlit/
    └── secrets.toml       # API keys (not in git)
```

## 🔧 Configuration

- **Clustering Parameters**: Adjust the number of clusters (2-10) in the UI
- **AI Model**: Default is `llama-3.3-70b-versatile` (configurable in app.py)
- **Theme**: Dark mode enabled by default

## 🤝 Team: rowlet-regression

Built for the Champions Group Datathon – turning raw data into actionable intelligence.

## 📝 License

This project is developed for educational purposes as part of NUS Datathon 2026.
