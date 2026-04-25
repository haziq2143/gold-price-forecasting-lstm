# Gold Price Prediction Analysis using Multivariate LSTM 🪙📈

This project focuses on analyzing and forecasting global gold prices by integrating multiple macroeconomic indicators. We utilize a Deep Learning approach with **Long Short-Term Memory (LSTM)** networks to capture long-term temporal dependencies in financial data.

## 👥 Group 5 Team
- **Its.hazen** - Web Developer & AI Lead
- [Member Name 1]
- [Member Name 2]

## 📋 Project Overview
Unlike traditional univariate models, this project implements a **Multivariate** approach. The model's predictions are influenced not just by past gold prices, but by a correlated set of economic drivers:
* **Gold Price**: Historical daily closing prices.
* **DXY (Dollar Index)**: Measuring the value of the USD.
* **FEDFUNDS**: Federal Reserve interest rates.
* **CPI (Consumer Price Index)**: A primary measure of inflation.
* **PAYEMS (Non-Farm Payrolls)**: US employment data.

## 🛠️ Technical Stack
* **Language:** Python 3.10+
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

## 📊 Model Performance & Results
The model was optimized using **Early Stopping** and **Learning Rate Reduction** to ensure the highest accuracy without overfitting.

* **MAE (Mean Absolute Error):** 77.92
* **Accuracy:** ~98.4% 
* **Training Period:** Stopped at Epoch 63 (Optimal convergence)

### Prediction Visualization
![Prediction Results](reports/figures/prediction_plot.png)
> The plot above shows the successful backtesting (Red vs Blue) and the 30-day future forecasting (Green line).

## 🚀 Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/its-hazen/Gold-Price-Analysis-LSTM.git](https://github.com/its-hazen/Gold-Price-Analysis-LSTM.git)
