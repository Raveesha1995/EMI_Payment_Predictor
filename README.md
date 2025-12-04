# EMI Payment Predictor - AI-Powered Payment Date Prediction System

An intelligent system combining **Machine Learning (XGBoost)** and **Large Language Models (LLM)** to predict EMI payment dates with high accuracy and provide AI-powered explanations.

## 🚀 Key Features

- **ML Prediction Engine**: XGBoost model analyzing payment patterns, delays, and trends
- **LLM-Powered Explanations**: OpenAI GPT-4 Turbo for intelligent, concise prediction explanations
- **AI Business Insights**: LLM-generated risk assessments and actionable recommendations
- **Smart Date Calculations**: Last Demand Date, Next Demand Date, Predicted Date
- **Modern Web Interface**: User-friendly UI with AI-powered explanations
- **RESTful API**: Flask backend with integrated LLM capabilities
- **CSV Export**: Generate comprehensive reports

## 📋 Requirements

- Python 3.8+
- OpenAI API Key (Required - Get from https://platform.openai.com/api-keys)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Raveesha1995/EMI_Payment_Predictor.git
   cd EMI_Payment_Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```
   Or set environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## 🚀 Quick Start

1. **Start Backend Server** (Terminal 1)
   ```bash
   python backend/run_backend.py
   ```
   Backend runs on: http://localhost:5000

2. **Start Frontend Server** (Terminal 2)
   ```bash
   python frontend/run_frontend.py
   ```
   Frontend runs on: http://localhost:3000 (opens automatically)

3. **Access the Application**
   - Open http://localhost:3000 in your browser
   - Enter a customer ID (e.g., CUST_0001)
   - Get AI-powered predictions with LLM explanations

## 📊 Project Structure

```
EMI_Payment_Predictor/
├── backend/              # Flask API server
│   ├── app.py           # API endpoints with LLM integration
│   └── run_backend.py   # Server startup
├── frontend/            # Web interface
│   ├── index.html       # Main UI
│   ├── app.js           # Frontend logic
│   ├── styles.css       # Styling
│   └── run_frontend.py  # Frontend server
├── data/                # Data files
│   └── emi_history.csv  # Payment history
├── models/              # ML models
│   └── emi_predictor_model.pkl
├── predictor.py         # ML prediction engine
├── llm_explainer.py     # LLM integration (Core feature)
├── data_processor.py    # Data processing
├── config.py            # Configuration
└── requirements.txt     # Dependencies
```

## 🔑 API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Single prediction with LLM explanation
- `POST /api/predict/batch` - Batch predictions with LLM insights
- `GET /api/customers` - List all customers
- `GET /api/customer/<id>/history` - Customer payment history
- `POST /api/train` - Train the ML model

## 💡 Tech Stack

- **Backend**: Python, Flask, XGBoost, scikit-learn
- **LLM**: OpenAI GPT-4 Turbo
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy

## 📝 License

MIT License

