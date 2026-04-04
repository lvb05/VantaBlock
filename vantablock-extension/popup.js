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
    const detection = result.lastDetection;
    
    if (detection && detection.verdict) {
      const v = detection.verdict.toUpperCase();
      verdictDiv.innerHTML = v === 'BOT' ? '🤖 BOT' : (v === 'HUMAN' ? '👤 HUMAN' : '❓ UNKNOWN');
      verdictDiv.className = `verdict ${detection.verdict}`;
      const conf = detection.confidence ? Math.round(detection.confidence * 100) : '?';
      confidenceDiv.innerHTML = `Confidence: ${conf}%`;
    } else {
      verdictDiv.innerHTML = '—';
      verdictDiv.className = 'verdict unknown';
      confidenceDiv.innerHTML = 'No data yet';
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