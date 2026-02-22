# ⚡ Power Consumption Forecasting System

This project analyzes historical electricity consumption data using time-series analysis and machine learning techniques.  
It applies **data preprocessing, regression modeling, and visualization methods** to identify consumption trends and forecast future power usage.

The project includes:
- 📊 Data cleaning and exploratory data analysis  
- 🤖 Machine learning forecasting model  
- 📈 Power BI dashboard for interactive insights  
- 📑 Presentation summarizing findings  

---

## 📌 Objective

The primary objective of this project is to:

- Analyze historical electricity consumption patterns  
- Identify seasonal and temporal trends  
- Forecast future power usage using machine learning  
- Build an interactive dashboard for analytical insights  

---

## 📊 Dataset

- **Source:** Historical Power Consumption Dataset  
- **Coverage:** Time-based electricity usage records  
- **Type:** Time-series structured dataset  

### Key Features:
- Date  
- Power Consumption  
- Month (Derived Feature)  
- Year (Derived Feature)  
- Additional time-based attributes  

### Files:
- `powerconsumption.csv` → Raw dataset  
- Processed dataset generated during preprocessing  

---

## 🛠️ Technologies Used

- **Python**  
- **Pandas, NumPy** → Data preprocessing  
- **Matplotlib, Seaborn** → Exploratory Data Analysis  
- **Scikit-learn** → Forecasting model development  
- **Power BI** → Dashboard visualization  
- **Pickle** → Model serialization  

---

## 🔄 Methodology

### 1️⃣ Data Preprocessing  
- Removed missing and inconsistent records  
- Converted date column into datetime format  
- Extracted year and month features  
- Structured dataset for modeling  

Notebook: `model.ipynb`

---

### 2️⃣ Exploratory Data Analysis (EDA)  
- Monthly consumption trend analysis  
- Year-wise comparison  
- Seasonal variation detection  
- Visualization of distribution patterns  

---

### 3️⃣ Forecasting Model  
- Performed train-test split  
- Applied regression-based machine learning model  
- Evaluated model performance  
- Generated future consumption predictions  

Notebook: `model.ipynb`  
Model saved as: `model.pkl`

---

## 📊 Dashboard

Power BI Dashboard: `power consumption dashboard.pbix`

Includes:
- Monthly and yearly consumption trends  
- Forecast visualization  
- Comparative analysis  
- Interactive filtering options  

---

## 📁 Project Structure

Power-Consumption-Forecaster/
- powerconsumption.csv
- model.ipynb
- app.py
- requirements.txt
- power consumption dashboard.pbix
- Power Consumption Forecast.pptx
- README.md

🔗 Project link: [Power-Consumption-Forecaster](https://github.com/pritam2005das/Power-Consumption-Forecaster)  

---

## 📈 Key Findings

- Electricity consumption exhibits clear seasonal patterns  
- Certain months consistently show higher usage  
- Time-based feature engineering improves forecasting accuracy  
- The regression model effectively captures consumption trends  

---

## 🔮 Future Scope

- Implement advanced time-series models (ARIMA, Prophet, LSTM)  
- Integrate weather and temperature data  
- Deploy a web-based real-time forecasting system  
- Add explainable AI techniques for better interpretability  

---

## 👨‍💻 Author

Pritam Das  
GitHub: https://github.com/pritam2005das
