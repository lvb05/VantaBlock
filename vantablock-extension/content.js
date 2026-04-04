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
const MAX_SEGMENT_EVENTS = 100;
const SEND_INTERVAL_MS = 2000;

let currentSegment = [];
let mousePathBuffer = [];
let lastMove = 0;
let lastPathFlush = 0;
let lastScrollY = window.scrollY;

function getTargetMeta(x, y) {
  const element = document.elementFromPoint(x, y);
  if (!element) {
    return { target_element: null, target_bbox: null };
  }

  const rect = element.getBoundingClientRect();
  const targetName = element.id || element.getAttribute('data-testid') || element.tagName.toLowerCase();

  return {
    target_element: targetName,
    target_bbox: {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height
    }
  };
}

function flushMousePath() {
  if (mousePathBuffer.length < 2) {
    mousePathBuffer = [];
    return;
  }

  const lastPoint = mousePathBuffer[mousePathBuffer.length - 1];
  const meta = getTargetMeta(lastPoint.x, lastPoint.y);

  currentSegment.push({
    type: 'mousemove',
    timestamp_ms: Date.now(),
    data: {
      path: mousePathBuffer,
      target_element: meta.target_element,
      target_bbox: meta.target_bbox
    }
  });

  mousePathBuffer = [];
  lastPathFlush = Date.now();
}

document.addEventListener('mousemove', (e) => {
  const now = Date.now();
  if (now - lastMove > 50) {
    mousePathBuffer.push({ x: e.clientX, y: e.clientY, t_ms: now });
    lastMove = now;
  }

  if (mousePathBuffer.length >= 6 || now - lastPathFlush > 200) {
    flushMousePath();
  }
});

document.addEventListener('click', (e) => {
  flushMousePath();
  const meta = getTargetMeta(e.clientX, e.clientY);
  currentSegment.push({
    type: 'click',
    timestamp_ms: Date.now(),
    data: {
      x: e.clientX,
      y: e.clientY,
      target_element: meta.target_element,
      target_bbox: meta.target_bbox
    }
  });
});

document.addEventListener('scroll', () => {
  flushMousePath();
  const now = Date.now();
  const deltaY = window.scrollY - lastScrollY;
  lastScrollY = window.scrollY;
  currentSegment.push({
    type: 'scroll',
    timestamp_ms: now,
    data: {
      deltaX: 0,
      deltaY: deltaY
    }
  });
});

document.addEventListener('keydown', (e) => {
  flushMousePath();
  currentSegment.push({
    type: 'keydown',
    timestamp_ms: Date.now(),
    data: {
      key: e.key
    }
  });
});

// ---------------------------
// 3. Send data to backend
// ---------------------------
let history = [];

async function sendSegment(events) {
  try {
    const response = await fetch('http://localhost:8001/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ events })
    });

    const result = await response.json();
    console.log('Prediction:', result);
    handleVerdict(result);
  } catch (err) {
    console.error('Backend error:', err);
  }
}

function handleVerdict(result) {
  if (typeof result?.bot_prob !== 'number') {
    return;
  }

  history.push(result.bot_prob);
  if (history.length > 3) {
    history.shift();
  }

  const avg = history.reduce((a, b) => a + b, 0) / history.length;
  console.log('Avg bot prob:', avg);

  if (avg > 0.8) {
    blockUser();
  }
}

function flushSegment(reason) {
  flushMousePath();
  if (currentSegment.length === 0) {
    return;
  }

  const segment = currentSegment;
  currentSegment = [];
  sendSegment(segment);
}

// ---------------------------
// 4. BLOCKING ACTIONS
// ---------------------------
function blockUser() {
  console.log('VantaBlock: blocking bot');
  document.body.innerHTML = `
    <div style="
      position: fixed;
      width: 100vw;
      height: 100vh;
      background: black;
      color: red;
      display: flex;
      justify-content: center;
      align-items: center;
      font-size: 24px;
      z-index: 999999;">
      Unauthorized agent detected
    </div>
  `;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'BLOCK') {
    blockUser();
    sendResponse({ status: 'blocked' });
  }
});

window.addEventListener('load', () => {
  console.log('Page loaded, starting prediction loop');
  setInterval(() => {
    if (currentSegment.length >= MAX_SEGMENT_EVENTS) {
      flushSegment('size');
    }
  }, 250);

  setInterval(() => {
    if (currentSegment.length > 0) {
      flushSegment('interval');
    }
  }, SEND_INTERVAL_MS);
});