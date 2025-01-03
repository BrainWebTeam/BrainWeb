import React from 'react';
import CodeBlock from '../../../CodeBlock';

function YahooFinanceTools() {
  return (
    <section id="yahoo-finance-tools">
      <h2>Yahoo Finance Tools</h2>
      <p>
        Tools for fetching and analyzing financial data from Yahoo Finance, including
        stock quotes, historical data, and financial statements.
      </p>

      <h3>Market Data</h3>
      <CodeBlock
        language="typescript"
        code={`interface MarketDataConfig {
  symbols: string[];
  range: 'live' | '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y' | '5y' | 'max';
  interval: '1m' | '5m' | '15m' | '30m' | '1h' | '1d' | '1wk' | '1mo';
  fields: {
    price: boolean;
    volume: boolean;
    marketCap: boolean;
    peRatio: boolean;
    custom: string[];
  };
  cache: {
    enabled: boolean;
    ttl: number;
  };
}

class MarketDataFetcher {
  async fetchMarketData(config: MarketDataConfig) {
    // Validate symbols
    await this.validateSymbols(config.symbols);
    
    // Check cache if enabled
    if (config.cache.enabled) {
      const cached = await this.checkCache(
        config.symbols,
        config.range
      );
      if (cached) return cached;
    }
    
    // Fetch data from Yahoo Finance
    const data = await Promise.all(
      config.symbols.map(symbol =>
        this.fetchSymbolData(symbol, config)
      )
    );
    
    // Update cache
    if (config.cache.enabled) {
      await this.updateCache(data, config.cache.ttl);
    }
    
    return data;
  }
}`}
      />

      <h3>Financial Analysis</h3>
      <CodeBlock
        language="typescript"
        code={`interface AnalysisConfig {
  statements: {
    income: boolean;
    balance: boolean;
    cashflow: boolean;
    frequency: 'annual' | 'quarterly';
  };
  metrics: {
    profitability: boolean;
    liquidity: boolean;
    solvency: boolean;
    valuation: boolean;
    custom: string[];
  };
  comparison: {
    enabled: boolean;
    peers: string[];
    benchmark: string;
  };
}

class FinancialAnalyzer {
  async analyzeFinancials(
    symbol: string,
    config: AnalysisConfig
  ) {
    // Fetch financial statements
    const statements = await this.fetchStatements(
      symbol,
      config.statements
    );
    
    // Calculate financial metrics
    const metrics = await this.calculateMetrics(
      statements,
      config.metrics
    );
    
    // Perform peer comparison if enabled
    if (config.comparison.enabled) {
      const comparison = await this.compareToPeers(
        metrics,
        config.comparison
      );
      return { metrics, comparison };
    }
    
    return { metrics };
  }
}`}
      />

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h4 className="text-lg font-semibold mb-4">Usage Notes</h4>
        <ul className="space-y-2">
          <li>Implement rate limiting to comply with API restrictions</li>
          <li>Cache frequently accessed data to improve performance</li>
          <li>Handle market hours and trading days appropriately</li>
          <li>Validate all input symbols before making requests</li>
          <li>Implement proper error handling for API failures</li>
        </ul>
      </div>
    </section>
  );
}

export default YahooFinanceTools;