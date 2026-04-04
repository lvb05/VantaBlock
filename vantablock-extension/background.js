chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'DETECT') {
    console.log('Mock mode: always blocking as bot');
    // Always send BLOCK command
    chrome.tabs.sendMessage(sender.tab.id, { type: 'BLOCK' });
    sendResponse({ verdict: 'bot' });
    return true;
  }
});