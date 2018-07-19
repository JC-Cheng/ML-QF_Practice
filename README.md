# Machine Learning & Quantitative Finance
Implementation of ML/QF algorithms. Feel free to [email me](mailto:jenchieh_cheng@mfe.berkeley.edu) if you have any comment or feedback.

## 1. Completed

- Deep Mixture Density Networks: deep MDN model implemented by Tensorflow
- Dynamic Fama-French with Kalman filter: recovering latent alpha & factor loadings based on observed returns
- Multi-Asset Momentum: extracting momentum signals from various asset class and test with different strategies
	- 1) Adaptive weights based on signal strength (softmax activation)
	- 2) Long/short mean-variance portfolio
	- 3) Long/short Beta neutral portfolio
- PCA Color Augmentation: a data augmentation technique widely used in image recognition
- Quantile Regression: estimating q% quantile of asset return
- Recommendation System: 
	- 1) similiarity-based recommender
	- 2) collaborative filitering (Tensorflow implementation)
- RNN Queue Imbalance: predicting next timestamp bid/ask direction with limit order book status, with LSTM model implemented in Keras

## 2. Ongoing

- Fractional Differencing on Factor Returns
- Python Trading Framework: a framework allowed to backtest trading strategies and evaluate performance metrics

## 3. Embryo

- Black-Litterman and Pair Trading