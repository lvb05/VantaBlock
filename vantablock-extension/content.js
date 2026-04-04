console.log('🟢 VantaBlock content script loaded on:', window.location.href);

// ---------------------------
// 1. Collect environment & automation signals
// ---------------------------
function getAutomationSignals() {
  const signals = {
    webdriver: navigator.webdriver === true,
    userAgent: navigator.userAgent,
    languages: navigator.languages,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: navigator.deviceMemory,
    pluginsLength: navigator.plugins.length
  };

  const botGlobals = [];
  if (window.cdc_adoQpoasnfa76pfcZLmcfl_Array) botGlobals.push('selenium');
  if (window.__webdriver_evaluate) botGlobals.push('webdriver_evaluate');
  if (window.__webdriver_script_function) botGlobals.push('webdriver_script');
  if (window.callPhantom) botGlobals.push('phantomjs');
  if (window._phantom) botGlobals.push('phantomjs_alt');
  signals.botGlobals = botGlobals;

  return signals;
}

// ---------------------------
// 2. Behavioural data collection
// ---------------------------
let mouseMovements = [];
let clicks = [];
let scrolls = [];

let lastMove = 0;
document.addEventListener('mousemove', (e) => {
  const now = Date.now();
  if (now - lastMove > 50) {
    mouseMovements.push({ x: e.clientX, y: e.clientY, t: now });
    lastMove = now;
  }
});

document.addEventListener('click', (e) => {
  clicks.push({ x: e.clientX, y: e.clientY, t: Date.now() });
});

document.addEventListener('scroll', () => {
  scrolls.push({ y: window.scrollY, t: Date.now() });
});

// ---------------------------
// 3. Send data to background
// ---------------------------
function sendForAnalysis() {
  console.log(' VantaBlock: sending data to background');
  const payload = {
    automation: getAutomationSignals(),
    mouse: mouseMovements,
    clicks: clicks,
    scrolls: scrolls,
    url: window.location.href,
    timestamp: Date.now()
  };
  console.log('Payload:', payload);
  chrome.runtime.sendMessage({ type: 'DETECT', data: payload });
}

// ---------------------------
// 4. BLOCKING ACTIONS
// ---------------------------
function blockBot() {
  console.log('🚫 VantaBlock: blocking bot');
  const blocker = document.createElement('div');
  blocker.id = 'vantablock-blocker';
  blocker.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.85);
    color: white;
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: Arial, sans-serif;
    font-size: 24px;
    text-align: center;
    pointer-events: all;
  `;
  blocker.innerHTML = `
    <div style="background: #d32f2f; padding: 30px; border-radius: 10px;">
        <strong>Access Denied</strong><br/>
      Automated activity detected. Session terminated.
    </div>
  `;
  document.body.appendChild(blocker);
  document.body.style.pointerEvents = 'none';
  document.querySelectorAll('input, textarea, button, a').forEach(el => {
    el.disabled = true;
    el.style.pointerEvents = 'none';
  });
  localStorage.clear();
  sessionStorage.clear();
  document.cookie.split(";").forEach(c => {
    document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
  });
  chrome.runtime.sendMessage({ type: 'CLEAR_COOKIES', url: window.location.origin });
  const logoutPaths = ['/logout', '/signout', '/logoff', '/exit', '/user/logout'];
  const currentOrigin = window.location.origin;
  for (let path of logoutPaths) {
    window.location.href = currentOrigin + path;
    break;
  }
  setTimeout(() => {
    if (window.location.href.indexOf('/logout') === -1) {
      window.location.reload();
    }
  }, 500);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'BLOCK') {
    blockBot();
    sendResponse({ status: 'blocked' });
  }
});

window.addEventListener('load', () => {
  console.log('📄 Page loaded, scheduling analysis in 3 seconds');
  setTimeout(sendForAnalysis, 3000);
});