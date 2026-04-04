const path = require('path');
const { chromium } = require('playwright');

async function main() {
  const extensionPath = path.resolve(__dirname, '..', 'vantablock-extension');
  const userDataDir = path.resolve(__dirname, '.user-data');
  const targetUrl = process.env.TARGET_URL || 'http://localhost:8001/demo/index.html';

  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: false,
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`
    ]
  });

  const page = await context.newPage();
  await page.goto(targetUrl);

  console.log('Extension loaded at:', extensionPath);
  console.log('Navigated to:', targetUrl);
}

main().catch((error) => {
  console.error('Failed to launch Playwright with extension:', error);
  process.exitCode = 1;
});
