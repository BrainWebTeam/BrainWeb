import React from 'react';
import CodeBlock from '../../../CodeBlock';

function WebTools() {
  return (
    <section id="web-tools">
      <h2>Web Tools</h2>
      <p>
        Tools for web scraping, content extraction, and web API integration.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface WebToolsConfig {
  scraping: {
    selector: string;
    waitFor?: string;
    timeout: number;
    proxy?: string;
  };
  extraction: {
    elements: string[];
    attributes: string[];
    text: boolean;
  };
  api: {
    endpoint: string;
    method: string;
    headers: Record<string, string>;
  };
}

class WebTools {
  async scrapeContent(url: string, config: WebToolsConfig) {
    // Initialize scraper
    const browser = await this.initBrowser();
    
    try {
      // Navigate to page
      const page = await browser.newPage();
      await page.goto(url, { timeout: config.timeout });
      
      // Extract content
      const content = await page.evaluate(
        (config) => {
          const elements = document.querySelectorAll(config.selector);
          return Array.from(elements).map(el => ({
            text: el.textContent,
            html: el.innerHTML,
            attributes: Object.fromEntries(
              config.attributes.map(attr => [
                attr, 
                el.getAttribute(attr)
              ])
            )
          }));
        },
        config
      );
      
      return content;
      
    } finally {
      await browser.close();
    }
  }
}`}
      />
    </section>
  );
}

export default WebTools;