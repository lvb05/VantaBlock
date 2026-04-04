const RULE_ID_OFFSET = 100000;

function blockRequestsForTab(tabId) {
  if (!chrome.declarativeNetRequest) {
    console.warn('VantaBlock: declarativeNetRequest not available');
    return;
  }

  const ruleId = RULE_ID_OFFSET + tabId;
  const rule = {
    id: ruleId,
    priority: 1,
    action: { type: 'block' },
    condition: {
      urlFilter: '*',
      resourceTypes: [
        'main_frame',
        'sub_frame',
        'xmlhttprequest',
        'fetch',
        'script',
        'image',
        'media',
        'font',
        'stylesheet',
        'other'
      ],
      tabIds: [tabId]
    }
  };

  chrome.declarativeNetRequest.updateDynamicRules(
    {
      removeRuleIds: [ruleId],
      addRules: [rule]
    },
    () => {
      if (chrome.runtime.lastError) {
        console.warn('VantaBlock: failed to add block rule', chrome.runtime.lastError.message);
      } else {
        console.log('VantaBlock: blocking requests for tab', tabId);
      }
    }
  );
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'DETECT') {
    console.log('Mock mode: always blocking as bot');
    if (sender.tab?.id != null) {
      chrome.tabs.sendMessage(sender.tab.id, { type: 'BLOCK' });
    }
    sendResponse({ verdict: 'bot' });
    return true;
  }

  if (message.type === 'BLOCK_REQUESTS') {
    if (sender.tab?.id != null) {
      blockRequestsForTab(sender.tab.id);
    }
  }

  if (message.type === 'KILL_TAB') {
    if (sender.tab?.id != null) {
      chrome.tabs.remove(sender.tab.id);
    }
  }
});