## Problem
Predict next-day stock price direction (up/down) using technical indicators.

## Baseline & bug found
Initial models (RF, GB, KNN, XGBoost) using RSI/MACD/Bollinger Bands scored 
~50% accuracy. Diagnosed a class-collapse issue: GridSearchCV's default 
scoring optimized only for the positive class, causing models to just 
predict "up" almost always. Fixed with scoring='f1_macro' + scale_pos_weight.

## Hypothesis & fix
After fixing the collapse, honest accuracy was ~48% — near-random, 
suggesting next-day direction is close to a random walk given only a 
stock's own technicals. Added Nifty 50 market-return features, hypothesizing 
broad market movement carries more signal than individual stock indicators.

## Features Used

| Feature | What it measures |
|---|---|
| RSI | Overbought/oversold level (0-100 scale) based on recent price moves |
| MACD | Difference between short-term and long-term price trend (momentum) |
| MACD_signal | Smoothed moving average of MACD, used to spot trend shifts |
| MACD_diff | MACD minus MACD_signal; a zero-crossing signals momentum change |
| BB_high / BB_low | Volatility-based price bands above/below the moving average |
| BB_width | Gap between BB_high and BB_low — wide = high volatility |
| return_1d / return_5d | Stock's own % price change over last 1 / 5 days |
| volatility_5d / volatility_10d | Std. deviation of returns over last 5 / 10 days (choppiness) |
| nifty_return_1d / nifty_return_5d | Broad market (Nifty 50) % return over last 1 / 5 days |

*Note: `nifty_return_1d` ranked as the single most important feature, 
supporting the finding that broad market movement predicts individual 
stock direction better than the stock's own technical indicators.*

## Result
Accuracy improved from 0.48 → 0.55. Nifty_return_1d became the top-ranked 
feature by importance, confirming the hypothesis. Confusion matrix stayed 
balanced across both classes (no collapse).