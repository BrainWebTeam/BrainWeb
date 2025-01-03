import { NodeContent } from '../../../../types/content';

export const timeSeriesContent: NodeContent = {
  title: 'Time Series Analysis',
  description: 'Methods and techniques for analyzing time-ordered data to extract meaningful patterns and predict future values.',
  concepts: [
    'ARIMA Models',
    'Seasonality',
    'Trend Analysis',
    'Forecasting',
    'Time Series Decomposition'
  ],
  examples: [
    {
      language: 'python',
      description: 'ARIMA model implementation',
      code: `from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Prepare time series data
data = pd.read_csv('timeseries.csv', parse_dates=['date'])
data.set_index('date', inplace=True)

# Fit ARIMA model
model = ARIMA(data, order=(1, 1, 1))
results = model.fit()

# Make predictions
forecast = results.forecast(steps=30)
print("Forecast:", forecast)`
    }
  ],
  resources: [
    {
      title: 'Time Series Guide',
      description: 'Comprehensive guide to time series analysis',
      url: 'https://otexts.com/fpp2/'
    },
    {
      title: 'Prophet Documentation',
      description: 'Facebook\'s time series forecasting tool',
      url: 'https://facebook.github.io/prophet/'
    }
  ],
  prerequisites: ['Statistics', 'Probability Theory', 'Python Programming'],
  relatedTopics: ['Forecasting', 'Signal Processing', 'Sequential Data']
};