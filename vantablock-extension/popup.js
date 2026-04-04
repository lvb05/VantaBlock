// Helper to get stored stats or initialize
function getStats() {
  return new Promise((resolve) => {
    chrome.storage.local.get(['totalChecks', 'blockedCount', 'humanCount'], (result) => {
      resolve({
        totalChecks: result.totalChecks || 0,
        blockedCount: result.blockedCount || 0,
        humanCount: result.humanCount || 0
      });
    });
  });
}

function updateStats(verdict) {
  getStats().then(stats => {
    let newStats = { ...stats };
    newStats.totalChecks += 1;
    if (verdict === 'bot') newStats.blockedCount += 1;
    if (verdict === 'human') newStats.humanCount += 1;
    chrome.storage.local.set(newStats);
    displayStats(newStats);
  });
}

function displayStats(stats) {
  document.getElementById('totalChecks').innerText = stats.totalChecks;
  document.getElementById('blockedCount').innerText = stats.blockedCount;
  document.getElementById('humanCount').innerText = stats.humanCount;
}

function updateUI() {
  chrome.storage.local.get(['lastDetection'], (result) => {
    const verdictDiv = document.getElementById('verdictText');
    const confidenceDiv = document.getElementById('confidenceText');
    const statusCard = document.getElementById('statusCard');
    const detection = result.lastDetection;

    if (detection && detection.verdict) {
      const v = detection.verdict.toLowerCase();
      const verdict = v === 'bot' ? 'bot' : (v === 'human' ? 'human' : 'unknown');
      const conf = detection.confidence ? Math.round(detection.confidence * 100) : '?';

      // Update verdict text with emoji
      if (verdict === 'bot') {
        verdictDiv.innerHTML = '🔴 Bot Detected';
      } else if (verdict === 'human') {
        verdictDiv.innerHTML = '🟢 Human';
      } else {
        verdictDiv.innerHTML = '🟡 Analyzing';
      }

      // Apply class for dynamic coloring
      statusCard.className = `status-card ${verdict}`;
      verdictDiv.className = `verdict ${verdict}`;

      // Show confidence with better messaging
      if (verdict === 'bot') {
        confidenceDiv.innerHTML = `${conf}% confidence • behavior anomaly detected`;
      } else if (verdict === 'human') {
        confidenceDiv.innerHTML = `${conf}% confidence • recognized as human`;
      } else {
        confidenceDiv.innerHTML = `Analyzing activity patterns...`;
      }
    } else {
      verdictDiv.innerHTML = '🟡 Analyzing';
      statusCard.className = 'status-card unknown';
      verdictDiv.className = 'verdict unknown';
      confidenceDiv.innerHTML = 'Awaiting scan results';
    }
  });

  // Also display stats
  getStats().then(stats => displayStats(stats));
}

// Refresh button: re‑fetch last detection and stats
document.getElementById('refreshBtn').addEventListener('click', () => {
  const btn = document.getElementById('refreshBtn');
  btn.innerHTML = '<span class="spinner"></span> Refreshing...';
  updateUI();
  setTimeout(() => {
    btn.innerHTML = '<span>⟳</span> Refresh';
  }, 500);
});

// Listen for new detections from background to update stats automatically
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.lastDetection) {
    const newVerdict = changes.lastDetection.newValue?.verdict;
    if (newVerdict) updateStats(newVerdict);
    updateUI();
  }
  if (area === 'local' && (changes.totalChecks || changes.blockedCount || changes.humanCount)) {
    getStats().then(stats => displayStats(stats));
  }
});

// Initial load
updateUI();