// Google Sheets via Apps Script — ratings submission
// Replace APPS_SCRIPT_URL after Step 12 Google Sheets setup
const APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbwjK2zBw01GCwoDJcPmMtBlLzVSTcHKyrOpCjGplEtdtaaHwyO_F41vZ2MulOqLCUpa/exec';

async function submitRatings(ratings) {
  const payload = {
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    ratings
  };
  try {
    await fetch(APPS_SCRIPT_URL, { method: 'POST', body: JSON.stringify(payload) });
  } catch (e) {
    console.error('Sheets submission failed:', e);
  }
}

window.submitRatings = submitRatings;
