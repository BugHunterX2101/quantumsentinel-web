// QuantumSentinel — frontend application logic (vanilla JS, no build step).
const state = {
  token: localStorage.getItem('qs_token') || null,
  user: JSON.parse(localStorage.getItem('qs_user') || 'null'),
  beginner: localStorage.getItem('qs_beginner') !== 'false',
  meta: null,
  exchanges: {},           // key -> exchange info+status from /api/exchanges
  preferredExchanges: new Set(JSON.parse(localStorage.getItem('qs_exchanges') || 'null') || ['US']),
  activeExchangeFilter: null, // null = all watchlist, string = exchange key filter
  dashSearchQuery: '',
  lastSignals: {},
  lastSignalPrices: {},
  watchlist: new Set(JSON.parse(localStorage.getItem('qs_watchlist') || 'null') || []),
  pollTimer: null,
  signalSocket: null,
  signalReconnectMs: 1000,
  activeView: 'dashboard',
  tokenExpireTimer: null,
  _signalCache: (() => {
    try { return JSON.parse(localStorage.getItem('qs_signal_cache') || 'null'); } catch { return null; }
  })(),
  _lastEtag: null,         // ETag from last /api/signals/latest response
};

// ===========================================================================
// Loading bar
// ===========================================================================
const loadingBar = (() => {
  const el = document.getElementById('loading-bar');
  let pending = 0, hideTimer = null;
  return {
    start() {
      pending++;
      clearTimeout(hideTimer);
      el.classList.add('active');
      el.style.width = '65%';
    },
    done() {
      pending = Math.max(0, pending - 1);
      if (pending === 0) {
        el.style.width = '100%';
        hideTimer = setTimeout(() => { el.classList.remove('active'); el.style.width = '0%'; }, 350);
      }
    },
  };
})();

// ===========================================================================
// Toasts
// ===========================================================================
function toast(title, body, type = 'info', duration = 3800) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  // Error text can originate at the API boundary; never inject it as HTML.
  const titleEl = document.createElement('div');
  titleEl.className = 'toast-title'; titleEl.textContent = title;
  const bodyEl = document.createElement('div');
  bodyEl.className = 'toast-body'; bodyEl.textContent = body || '';
  el.append(titleEl, bodyEl);
  container.appendChild(el);
  setTimeout(() => {
    el.classList.add('leaving');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, duration);
}

// ===========================================================================
// API helper (wired to loading bar + toast on error)
// ===========================================================================
function api(path, opts = {}, opts2 = {}) {
  const headers = Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {});
  if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
  // ETag optimization: send If-None-Match only on GET signal requests (never on POST/refresh)
  const isGet = !opts.method || opts.method.toUpperCase() === 'GET';
  if (isGet && path === '/api/signals/latest' && state._lastEtag) {
    headers['If-None-Match'] = state._lastEtag;
  }
  loadingBar.start();
  return fetch(path, Object.assign({}, opts, { headers })).then(async (r) => {
    // 304 Not Modified — return cached data immediately (zero payload)
    if (r.status === 304 && state._signalCache) {
      loadingBar.done();
      return state._signalCache;
    }
    // Store ETag for next request
    const etag = r.headers.get('ETag');
    if (etag) state._lastEtag = etag;
    const body = await r.json().catch(() => ({}));
    if (r.status === 401) { handleTokenExpiry(); throw new Error('Session expired. Please log in again.'); }
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  }).catch((err) => {
    if (!opts2.silent) toast('Request failed', err.message, 'error');
    throw err;
  }).finally(() => loadingBar.done());
}

function handleTokenExpiry() {
  if (state.tokenExpireTimer) { clearTimeout(state.tokenExpireTimer); state.tokenExpireTimer = null; }
  if (state.signalSocket) { state.signalSocket.close(); state.signalSocket = null; }
  setLiveIndicator('disconnected');
  localStorage.removeItem('qs_token');
  localStorage.removeItem('qs_user');
  localStorage.removeItem('qs_login_at');
  localStorage.removeItem('qs_expires_in');
  state.token = null;
  state.user = null;
  document.getElementById('app').classList.add('hidden');
  document.getElementById('auth-screen').classList.remove('hidden');
  toast('Session expired', 'Please log in again.', 'error', 5000);
}

function scheduleTokenExpiry(expiresInSeconds) {
  if (state.tokenExpireTimer) clearTimeout(state.tokenExpireTimer);
  // Refresh warning 60 seconds before expiry (or immediately if < 60s left)
  const warnIn = Math.max(0, (expiresInSeconds - 60) * 1000);
  state.tokenExpireTimer = setTimeout(() => {
    toast('Session expiring soon', 'Your session will expire in 60 seconds. Please log out and log back in.', 'info', 8000);
    // Force logout at actual expiry
    setTimeout(handleTokenExpiry, 60000);
  }, warnIn);
}

const b64encode = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)));
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
}[char]));

// ===========================================================================
// Ripple effect on all buttons
// ===========================================================================
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.ripple-btn');
  if (!btn) return;
  const rect = btn.getBoundingClientRect();
  const ripple = document.createElement('span');
  const size = Math.max(rect.width, rect.height) * 1.4;
  ripple.className = 'ripple';
  ripple.style.width = ripple.style.height = size + 'px';
  ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
  ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
  btn.appendChild(ripple);
  setTimeout(() => ripple.remove(), 650);
});

// 2D canvas particle system removed — replaced by Three.js QS3D engine (bg3d.js).
// The QS3D engine renders to #auth-bg-canvas and #app-bg-canvas,
// initialised from the window 'load' handler below.

// ===========================================================================
// Typewriter tagline on auth screen
// ===========================================================================
(function typewriter() {
  const el = document.getElementById('tagline');
  const text = 'Open-source institutional terminal. FIPS 203/204 post-quantum session security.';
  let i = 0;
  function tick() {
    el.textContent = text.slice(0, i);
    i++;
    if (i <= text.length) setTimeout(tick, 22);
  }
  tick();
})();

// ===========================================================================
// Animated number counters
// ===========================================================================
function animateCounter(el, to, { duration = 900, decimals = 0, prefix = '', suffix = '' } = {}) {
  const from = parseFloat(el.dataset.rawValue || 0);
  const start = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = from + (to - from) * eased;
    el.textContent = prefix + val.toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(frame); else el.dataset.rawValue = to;
  }
  requestAnimationFrame(frame);
}

function animateScoreRing(score) {
  const circle = document.getElementById('score-ring-fill');
  const circumference = 2 * Math.PI * 18;
  const offset = circumference * (1 - score / 100);
  circle.style.strokeDashoffset = offset;
  let color = '#3ddc97';
  if (score < 60) color = '#ff5d7a'; else if (score < 90) color = '#ffd166';
  circle.style.stroke = color;
  document.getElementById('safety-score-badge').style.color = color;
  animateCounter(document.getElementById('safety-score-value'), score, { duration: 900, decimals: 0 });
}

// ===========================================================================
// Auth screen wiring
// ===========================================================================
document.querySelectorAll('.auth-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.auth-tab').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    const showLogin = btn.dataset.tab === 'login';
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    (showLogin ? registerForm : loginForm).classList.add('hidden');
    const target = showLogin ? loginForm : registerForm;
    target.classList.remove('hidden');
    target.style.animation = 'none';
    requestAnimationFrame(() => { target.style.animation = ''; });
  });
});

// ===========================================================================
// Password strength meter (register form)
// ===========================================================================
(function initPasswordStrength() {
  const input   = document.getElementById('register-password');
  const wrap    = document.querySelector('.pw-strength-wrap');
  const label   = document.getElementById('pw-strength-label');
  const fill    = document.getElementById('pw-strength-fill');
  const submitBtn = document.getElementById('register-submit');
  if (!input || !wrap) return;

  const RULES = [
    { id: 'req-len',   test: v => v.length >= 14 },
    { id: 'req-upper', test: v => /[A-Z]/.test(v) },
    { id: 'req-lower', test: v => /[a-z]/.test(v) },
    { id: 'req-digit', test: v => /\d/.test(v) },
    { id: 'req-sym',   test: v => /[!@#$%^&*()\-_=+\[\]{};:'",.<>?/\\|`~]/.test(v) },
    { id: 'req-rep',   test: v => !/(.)\1{3,}/.test(v) },
  ];

  const LEVELS = [
    { cls: '',   text: '' },
    { cls: 's1', text: 'WEAK' },
    { cls: 's2', text: 'FAIR' },
    { cls: 's3', text: 'STRONG' },
    { cls: 's4', text: 'VERY STRONG' },
  ];

  input.addEventListener('input', () => {
    const v = input.value;
    let passed = 0;
    RULES.forEach(rule => {
      const el = document.getElementById(rule.id);
      const ok = rule.test(v);
      if (ok) { passed++; el.className = 'pw-req met'; }
      else         { el.className = v.length > 0 ? 'pw-req fail' : 'pw-req'; }
    });

    // Map 0-6 passed rules → 4 strength levels
    let level = 0;
    if (v.length > 0) {
      if (passed <= 2) level = 1;
      else if (passed <= 4) level = 2;
      else if (passed === 5) level = 3;
      else level = 4;
    }

    // Update bar class
    wrap.classList.remove('s1', 's2', 's3', 's4');
    if (level > 0) wrap.classList.add(LEVELS[level].cls);
    label.textContent = LEVELS[level].text;

    // Gate submit button — all 6 rules must pass
    submitBtn.disabled = (passed < RULES.length);
  });
}());


function setButtonLoading(btn, loading, loadingText) {
  const labelEl = btn.querySelector('.btn-label');
  if (!labelEl) return; // guard: no label span, skip
  if (loading) {
    btn.dataset.originalLabel = labelEl.textContent;
    labelEl.innerHTML = `<span class="spinner"></span>${loadingText}`;
    btn.disabled = true;
  } else {
    labelEl.textContent = btn.dataset.originalLabel || '';
    btn.disabled = false;
  }
}

document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  const errEl = document.getElementById('login-error');
  const btn = e.target.querySelector('button[type=submit]');
  errEl.textContent = '';
  setButtonLoading(btn, true, 'Authenticating…');
  try {
    const data = await api('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }, { silent: true });
    await afterLogin(data);
  } catch (err) { errEl.textContent = err.message; } finally { setButtonLoading(btn, false); }
});

document.getElementById('register-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const email    = document.getElementById('register-email').value;
  const password = document.getElementById('register-password').value;
  const errEl    = document.getElementById('register-error');
  const btn      = document.getElementById('register-submit');
  const breachEl = document.getElementById('register-breach-warning');
  errEl.textContent = '';
  breachEl.classList.add('hidden');
  breachEl.textContent = '';
  setButtonLoading(btn, true, 'Generating ML-KEM-768 / ML-DSA-65 keys…');
  try {
    const regData = await api('/api/auth/register', { method: 'POST', body: JSON.stringify({ email, password }) }, { silent: true });
    // Show HIBP breach warning if server flagged it
    if (regData && regData.breach_warning) {
      breachEl.textContent = regData.breach_warning;
      breachEl.classList.remove('hidden');
      // Give user 4 seconds to read the warning before proceeding
      await new Promise(r => setTimeout(r, 4000));
    }
    const loginData = await api('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }, { silent: true });
    await afterLogin(loginData);
  } catch (err) { errEl.textContent = err.message; } finally { setButtonLoading(btn, false); }
});


async function afterLogin(data) {
  state.token = data.access_token;
  state.user = data.user;
  localStorage.setItem('qs_token', state.token);
  localStorage.setItem('qs_user', JSON.stringify(state.user));
  // Store login timestamp and TTL so page-reload can compute actual remaining time
  localStorage.setItem('qs_login_at', Date.now().toString());
  localStorage.setItem('qs_expires_in', String(data.expires_in || 900));
  // Schedule a warning before the JWT expires based on actual TTL from server
  scheduleTokenExpiry(data.expires_in || 900);
  document.getElementById('auth-screen').classList.add('hidden');
  // Switch 3D background: auth canvas out, app canvas in
  const authCanvas = document.getElementById('auth-bg-canvas');
  const appCanvas  = document.getElementById('app-bg-canvas');
  if (authCanvas) { authCanvas.style.opacity = '0'; setTimeout(() => { authCanvas.style.display = 'none'; }, 600); }
  if (appCanvas)  {
    appCanvas.style.display = '';
    // Use window.QS3D?.init to gracefully handle the case where bg3d.js hasn't
    // fully executed yet (e.g. slow network, script load race with defer attribute).
    setTimeout(() => window.QS3D?.init('app-bg-canvas', 'dashboard'), 100);
  }
  await performHandshake({ showOverlay: true });
  document.getElementById('app').classList.remove('hidden');
  document.getElementById('user-email').textContent = state.user.email;
  toast('Welcome', `Signed in as ${state.user.email}`, 'success');
  await bootstrapApp();
}

// ===========================================================================
// PQC hybrid handshake (real server-side ML-KEM-768 + ML-DSA-65 + X25519)
// ===========================================================================
async function performHandshake({ showOverlay = false } = {}) {
  const overlay = document.getElementById('handshake-overlay');
  const overlayText = document.getElementById('handshake-overlay-text');
  const steps = [
    'Generating X25519 ephemeral keypair…',
    'Encapsulating ML-KEM-768 shared secret (FIPS 203)…',
    'Deriving session key via HKDF-SHA256…',
    'Verifying ML-DSA-65 ServerHello signature (FIPS 204)…',
  ];
  let stepTimer;
  if (showOverlay) {
    overlay.classList.remove('hidden');
    let i = 0;
    overlayText.textContent = steps[0];
    stepTimer = setInterval(() => { i = (i + 1) % steps.length; overlayText.textContent = steps[i]; }, 420);
  }

  let clientPub = null;
  let usedRealWebCrypto = false;
  try {
    const kp = await crypto.subtle.generateKey({ name: 'X25519' }, true, ['deriveBits']);
    const rawPub = await crypto.subtle.exportKey('raw', kp.publicKey);
    clientPub = b64encode(rawPub);
    usedRealWebCrypto = true;
  } catch (e) {
    const rnd = new Uint8Array(32);
    crypto.getRandomValues(rnd);
    clientPub = b64encode(rnd);
  }
  const nonce = new Uint8Array(32);
  crypto.getRandomValues(nonce);
  const clientNonce = b64encode(nonce);

  const minDisplay = showOverlay ? new Promise((r) => setTimeout(r, 1400)) : Promise.resolve();
  const [result] = await Promise.all([
    api('/api/auth/pqc-handshake', {
      method: 'POST',
      body: JSON.stringify({ x25519_public_key: clientPub, client_nonce: clientNonce }),
    }),
    minDisplay,
  ]);

  if (showOverlay) {
    clearInterval(stepTimer);
    overlayText.textContent = '✓ Secure session established';
    await new Promise((r) => setTimeout(r, 450));
    overlay.classList.add('hidden');
  }

  state.handshake = Object.assign({}, result, { usedRealWebCrypto });
  renderHandshakeTrace();
}

function renderHandshakeTrace() {
  const el = document.getElementById('handshake-trace');
  if (!el || !state.handshake) return;
  const h = state.handshake;
  const rows = [
    ['Classical leg', h.usedRealWebCrypto ? 'Browser Web Crypto X25519 (real ECDH)' : 'X25519 fallback (browser lacks Web Crypto X25519)'],
    ['ML-KEM-768 encapsulation', h.kem_encapsulate_ms + ' ms (server, FIPS 203)'],
    ['ML-KEM-768 ciphertext size', h.algorithm_sizes.ml_kem_ciphertext_bytes + ' bytes (spec: 1088)'],
    ['ML-KEM-768 shared secret size', h.algorithm_sizes.ml_kem_shared_secret_bytes + ' bytes (spec: 32)'],
    ['ML-DSA-65 ServerHello signature', h.algorithm_sizes.ml_dsa_signature_bytes + ' bytes (spec: 3309)'],
    ['Derived session key', 'HKDF-SHA256 → ' + h.algorithm_sizes.session_key_bytes + ' bytes'],
    ['Session ID', h.session_id],
    ['Client ML-KEM keypair', h.simulated_client_kem_keypair ? 'server-generated demo keypair (browser has no ML-KEM)' : 'client-supplied'],
  ];
  el.innerHTML = rows.map(([label, val], i) =>
    `<div class="handshake-step" style="animation-delay:${i * 70}ms"><span class="label">${escapeHtml(String(label))}:</span><br><span class="val">${escapeHtml(String(val))}</span></div>`
  ).join('');
}

// ===========================================================================
// App shell: tabs, beginner mode, logout, polling
// ===========================================================================
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => switchView(btn.dataset.view));
});

// ── Keyboard shortcuts 1-7 ──────────────────────────────────────
const VIEW_KEYS = ['dashboard','trading','strategies','research','portfolio','security','integrations','community'];
document.addEventListener('keydown', (e) => {
  // Only when not typing in an input/textarea
  if (e.target.matches('input,textarea,select')) return;
  const idx = parseInt(e.key, 10) - 1;
  if (idx >= 0 && idx < VIEW_KEYS.length) switchView(VIEW_KEYS[idx]);
});

// ── Browser back/forward support ────────────────────────────────
window.addEventListener('popstate', () => {
  const hashView = location.hash.replace('#', '');
  if (VIEW_KEYS.includes(hashView) && hashView !== state.activeView) {
    switchView(hashView);
  }
});

// ── Confirmation modal helper ───────────────────────────────────
function confirmAction(title, body) {
  return new Promise((resolve) => {
    const modal  = document.getElementById('confirm-modal');
    const titleEl = document.getElementById('confirm-title');
    const bodyEl  = document.getElementById('confirm-body');
    const okBtn   = document.getElementById('confirm-ok');
    const cancelBtn = document.getElementById('confirm-cancel');
    titleEl.textContent = title;
    bodyEl.textContent  = body;
    modal.classList.remove('hidden');
    const cleanup = (result) => {
      modal.classList.add('hidden');
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      resolve(result);
    };
    const onOk     = () => cleanup(true);
    const onCancel = () => cleanup(false);
    okBtn.addEventListener('click', onOk, { once: true });
    cancelBtn.addEventListener('click', onCancel, { once: true });
  });
}

const PAGE_TITLES = { dashboard:'Dashboard', trading:'Order Desk', strategies:'Strategies', research:'Research',
  portfolio:'Portfolio', security:'Security', integrations:'Integrations', community:'Open Source' };

// Smart freshness gate — skip re-fetching data if the view was loaded < 5s ago
const _viewLastLoaded = {};
const _VIEW_FRESHNESS_MS = 5000;

function switchView(view) {
  state.activeView = view;
  // Update tab active state
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  // Update views with enter animation
  document.querySelectorAll('.view').forEach((v) => {
    const isTarget = v.id === 'view-' + view;
    v.classList.toggle('active', isTarget);
    if (isTarget) { v.classList.remove('entering'); requestAnimationFrame(() => v.classList.add('entering')); }
  });
  // Breadcrumb
  const titleEl = document.getElementById('page-title');
  if (titleEl) titleEl.textContent = PAGE_TITLES[view] || view;
  // URL hash — pushState on first visit (enables back-button), replaceState on revisit
  if (location.hash.replace('#', '') !== view) {
    history.pushState(null, '', '#' + view);
  } else {
    history.replaceState(null, '', '#' + view);
  }
  // Beginner banner
  const banners = {
    dashboard: 'Tip: BUY signals with higher confidence bars indicate stronger multi-asset agreement. Use keys 1–7 to navigate.',
    trading: 'Tip: every order is cryptographically signed with ML-DSA-65 before submission and settles as a paper (simulated) trade.',
    portfolio: 'Tip: Sharpe ratio > 1 is generally considered good risk-adjusted performance. Max drawdown shows worst peak-to-trough loss.',
    strategies: 'Tip: validate a strategy on historical data first. A positive backtest is not a prediction of future returns.',
    research: 'Tip: use walk-forward validation to detect overfitting. The Deflated Sharpe Ratio accounts for the number of strategies you tested.',
    security: 'Tip: the Quantum Safety Score reflects how fresh your cryptographic keys are — green means fully rotated and FIPS-compliant.',
  };
  const banner = document.getElementById('beginner-banner');
  if (state.beginner && banners[view]) { banner.textContent = banners[view]; banner.classList.remove('hidden'); }
  else banner.classList.add('hidden');

  // Smart data loading — skip if view was loaded < 5s ago to avoid redundant API calls
  const now = Date.now();
  const fresh = (_viewLastLoaded[view] && (now - _viewLastLoaded[view]) < _VIEW_FRESHNESS_MS);
  _viewLastLoaded[view] = now;

  if (!fresh) {
    if (view === 'dashboard')   loadDashboard();
    if (view === 'trading')     loadTrading();
    if (view === 'strategies')  loadStrategies();
    if (view === 'portfolio')   loadPortfolio();
    if (view === 'security')    loadSecurity();
    if (view === 'integrations') loadIntegrations();
    if (view === 'community')   loadCommunity();
  }
  restartPolling();
}

function restartPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  // community tab fetches from GitHub API — exclude from polling to avoid rate-limits
  const loaders = { dashboard: () => loadDashboard(true), trading: () => refreshOrders(true), strategies: loadStrategies, portfolio: () => loadPortfolio(true), security: () => loadSecurity(true), integrations: loadIntegrations };
  const fn = loaders[state.activeView];
  if (!fn) return;
  // Poll every 30s on dashboard (aligns with backend CACHE_TTL_SECONDS=30), 20s elsewhere
  const interval = state.activeView === 'dashboard' ? 30000 : 20000;
  state.pollTimer = setInterval(() => fn(true), interval);
}

function applyBeginnerMode() {
  document.querySelectorAll('[data-beginner]').forEach((el) => el.classList.toggle('hidden', !state.beginner));
  document.getElementById('beginner-state').textContent = state.beginner ? 'On' : 'Off';
}

document.getElementById('beginner-toggle').addEventListener('click', () => {
  state.beginner = !state.beginner;
  localStorage.setItem('qs_beginner', state.beginner);
  applyBeginnerMode();
  switchView(document.querySelector('.tab-btn.active').dataset.view);
});

document.getElementById('logout-btn').addEventListener('click', () => {
  if (state.signalSocket) state.signalSocket.close();
  if (state.tokenExpireTimer) clearTimeout(state.tokenExpireTimer);
  setLiveIndicator('disconnected');
  localStorage.removeItem('qs_token');
  localStorage.removeItem('qs_user');
  localStorage.removeItem('qs_login_at');
  localStorage.removeItem('qs_expires_in');
  location.reload();
});

// -- 20 preloaded assets — always available without a search -------------------
// Any OTHER ticker (e.g. RELIANCE.NS, BTC-USD, SAP.DE) can be searched in
// real time via the search bar and fetched live from Yahoo Finance.
const ASSET_GROUPS = [
  { label: 'Technology',        prefix: ['AAPL','MSFT','NVDA','GOOGL','META','AMD'] },
  { label: 'EV & Consumer',     prefix: ['TSLA','AMZN'] },
  { label: 'Financials',        prefix: ['JPM','V'] },
  { label: 'Healthcare',        prefix: ['JNJ','LLY'] },
  { label: 'Energy',            prefix: ['XOM'] },
  { label: 'ETFs',              prefix: ['SPY','QQQ'] },
  { label: 'Commodities',       prefix: ['GLD'] },
  { label: 'Crypto-Equity',     prefix: ['COIN'] },
  { label: 'Entertainment',     prefix: ['NFLX'] },
  { label: 'Travel',            prefix: ['BKNG'] },
  { label: 'Semiconductors',    prefix: ['TSM'] },
];

function buildAssetOptions(trackedAssets) {
  const set = new Set(trackedAssets);
  let html = '';
  for (const grp of ASSET_GROUPS) {
    const opts = grp.prefix.filter(t => set.has(t));
    if (!opts.length) continue;
    html += `<optgroup label="${escapeHtml(grp.label)}">`;
    html += opts.map(a => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join('');
    html += '</optgroup>';
    opts.forEach(t => set.delete(t));
  }
  // Any remaining not in groups
  if (set.size) {
    html += '<optgroup label="Other">';
    html += [...set].sort().map(a => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join('');
    html += '</optgroup>';
  }
  return html;
}

async function bootstrapApp() {
  applyBeginnerMode();

  // C6: Resilient meta fetch — a /api/meta failure must not block the whole app.
  // Graceful defaults let the user still trade and navigate, with degraded exchange data.
  const META_DEFAULTS = {
    tracked_assets: [
      ...ASSET_GROUPS.flatMap(g => g.prefix),
    ],
    exchanges: [],
    version: 'unknown',
  };
  try {
    state.meta = await api('/api/meta');
  } catch (err) {
    console.warn('bootstrapApp: /api/meta failed, using defaults:', err.message);
    state.meta = META_DEFAULTS;
  }
  // Populate exchange map from meta (avoids extra round-trip)
  if (state.meta.exchanges) {
    _exchangeData = state.meta.exchanges;
    state.exchanges = state.meta.exchanges;
  }
  const assetHtml = buildAssetOptions(state.meta.tracked_assets || META_DEFAULTS.tracked_assets);
  const sel = document.getElementById('order-asset');
  sel.innerHTML = assetHtml;
  document.getElementById('strategy-asset').innerHTML = assetHtml;
  // Restore view from URL hash
  const hashView = location.hash.replace('#', '');
  switchView(VIEW_KEYS.includes(hashView) ? hashView : 'dashboard');
  connectSignalStream();
  startOnboardingIfNeeded();
  sel.addEventListener('change', () => { updateOrderPricePreview(); updateAssetInfo(); });
  updateOrderPricePreview();
  updateAssetInfo();
  _startOrderPricePolling();
  const searchEl = document.getElementById('order-search');
  if (searchEl) searchEl.addEventListener('input', filterOrderList);
  startRefreshCountdown();
  // Load user preferences (watchlist + exchanges) in parallel
  api('/api/preferences', {}, { silent: true }).then(r => {
    if (r?.watchlist?.length) _saveWatchlistToStorage(r.watchlist);
    if (r?.preferred_exchanges?.length) {
      _saveExchangesToStorage(r.preferred_exchanges);
    }
    renderMarketStatusStrip();
    renderExchangeFilterPills();
  }).catch(() => {});
  // Initial market status strip render from meta data
  renderMarketStatusStrip();
  renderExchangeFilterPills();
}


function setLiveIndicator(status) {
  const dot = document.getElementById('live-indicator');
  if (!dot) return;
  dot.classList.remove('disconnected', 'reconnecting');
  const titles = { connected: 'Live — WebSocket connected', reconnecting: 'Reconnecting…', disconnected: 'Disconnected — will retry' };
  if (status === 'disconnected') dot.classList.add('disconnected');
  if (status === 'reconnecting') dot.classList.add('reconnecting');
  dot.title = titles[status] || '';
}

// ===========================================================================
// Exchange Manager — Global Market Selection + Market Hours
// ===========================================================================
const EXCHANGE_FLAGS = {
  US: '🇺🇸', NSE: '🇮🇳', LSE: '🇬🇧', XETRA: '🇩🇪',
  TSE: '🇯🇵', HKEX: '🇭🇰', ASX: '🇦🇺', TSX: '🇨🇦', CRYPTO: '₿',
};

let _exchangeData = {};     // fetched from /api/exchanges
let _pendingExchanges = null; // working set in modal

function _saveExchangesToStorage(list) {
  localStorage.setItem('qs_exchanges', JSON.stringify(list));
  state.preferredExchanges = new Set(list);
}

async function loadExchanges() {
  try {
    const data = await api('/api/exchanges', {}, { silent: true });
    _exchangeData = data;
    state.exchanges = data;
    renderMarketStatusStrip();
    renderExchangeFilterPills();
  } catch (_) {}
}

function renderMarketStatusStrip() {
  const strip = document.getElementById('market-status-strip');
  if (!strip) return;
  const preferred = [...state.preferredExchanges];
  if (!preferred.length || !Object.keys(_exchangeData).length) { strip.innerHTML = ''; return; }
  strip.innerHTML = preferred.map(key => {
    const ex = _exchangeData[key];
    if (!ex) return '';
    const st = ex.market_status?.status || 'closed';
    const label = ex.market_status?.label || 'CLOSED';
    const localTime = ex.market_status?.local_time || '';
    const flag = EXCHANGE_FLAGS[key] || '🌐';
    const isActive = state.activeExchangeFilter === key ? 'active-exch' : '';
    return `<div class="market-strip-item ${isActive}" data-exch="${escapeHtml(key)}" title="${escapeHtml(ex.name)} — ${escapeHtml(ex.tz)}">
      <span class="strip-dot ${st}"></span>
      <span>${flag} ${escapeHtml(key)}</span>
      <span style="color:var(--text-4);">${localTime}</span>
      <span style="font-weight:600;color:${st==='open'?'var(--green)':st==='pre'?'var(--amber)':'var(--text-4)'};">${label}</span>
    </div>`;
  }).join('');
  strip.querySelectorAll('.market-strip-item').forEach(el => {
    el.addEventListener('click', () => {
      const key = el.dataset.exch;
      if (state.activeExchangeFilter === key) {
        state.activeExchangeFilter = null;
      } else {
        state.activeExchangeFilter = key;
      }
      renderMarketStatusStrip();
      renderExchangeFilterPills();
      _applyDashboardFilter();
    });
  });
}

function renderExchangeFilterPills() {
  const container = document.getElementById('exchange-filter-pills');
  if (!container || !Object.keys(_exchangeData).length) return;
  const preferred = [...state.preferredExchanges];
  if (!preferred.length) { container.innerHTML = ''; return; }
  // "All" pill + one per preferred exchange
  const allActive = !state.activeExchangeFilter ? 'active' : '';
  let html = `<button class="exch-pill ${allActive}" data-exch="">ALL</button>`;
  for (const key of preferred) {
    const ex = _exchangeData[key];
    if (!ex) continue;
    const active = state.activeExchangeFilter === key ? 'active' : '';
    const flag = EXCHANGE_FLAGS[key] || '';
    html += `<button class="exch-pill ${active}" data-exch="${escapeHtml(key)}">${flag} ${escapeHtml(key)}</button>`;
  }
  container.innerHTML = html;
  container.querySelectorAll('.exch-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      state.activeExchangeFilter = btn.dataset.exch || null;
      renderExchangeFilterPills();
      renderMarketStatusStrip();
      _applyDashboardFilter();
    });
  });
}

// Instant in-page filter (no API call — O(n) DOM traversal)
function _applyDashboardFilter() {
  const query = (state.dashSearchQuery || '').toLowerCase().trim();
  const exchFilter = state.activeExchangeFilter;
  const assetExchMap = state.meta?.asset_exchange_map || {};
  const cards = document.querySelectorAll('#signal-grid .signal-card');
  let visible = 0;
  cards.forEach(card => {
    const ticker = card.id.replace('signal-', '');
    const matchesSearch = !query || ticker.toLowerCase().includes(query);
    const matchesExch = !exchFilter || (assetExchMap[ticker] || 'US') === exchFilter;
    if (matchesSearch && matchesExch) {
      card.style.display = '';
      visible++;
    } else {
      card.style.display = 'none';
    }
  });
  // Update count in meta strip
  const metaEl = document.getElementById('signal-meta');
  if (metaEl && (query || exchFilter)) {
    const existing = metaEl.querySelector('.filter-result-badge');
    if (existing) existing.remove();
    const badge = document.createElement('span');
    badge.className = 'filter-result-badge signal-count-chip';
    badge.style.marginLeft = '10px';
    badge.textContent = `${visible} result${visible !== 1 ? 's' : ''}`;
    metaEl.appendChild(badge);
  } else if (metaEl) {
    const existing = metaEl.querySelector('.filter-result-badge');
    if (existing) existing.remove();
  }
}

// Exchange modal
function openExchangeModal() {
  _pendingExchanges = new Set(state.preferredExchanges);
  _renderExchangeGrid();
  document.getElementById('exchange-modal').classList.remove('hidden');
}

function _renderExchangeGrid() {
  const grid = document.getElementById('exchange-grid');
  if (!grid || !Object.keys(_exchangeData).length) {
    if (grid) grid.innerHTML = '<div class="empty-state" style="padding:48px 24px;">Loading exchange data…</div>';
    return;
  }
  const infoEl = document.getElementById('exchange-selected-info');
  if (infoEl) infoEl.textContent = `${_pendingExchanges.size} exchange${_pendingExchanges.size !== 1 ? 's' : ''} selected`;

  grid.innerHTML = Object.entries(_exchangeData).map(([key, ex]) => {
    const selected = _pendingExchanges.has(key) ? 'selected' : '';
    const st = ex.market_status?.status || 'closed';
    const stLabel = ex.market_status?.label || 'CLOSED';
    const localTime = ex.market_status?.local_time || '';
    const assetCount = ex.asset_count || 0;
    const flag = EXCHANGE_FLAGS[key] || '🌐';
    const opensIn = ex.market_status?.opens_in_mins;
    const closesIn = ex.market_status?.closes_in_mins;
    // FIX F11: CRYPTO status is 'open' (like all open exchanges), NOT 'crypto'.
    // The old `st === 'crypto'` branch could never match any real status value.
    const timeHint = st === 'open' && closesIn != null ? `Closes in ${closesIn}m` :
                     st === 'pre' && opensIn != null ? `Opens in ${opensIn}m` :
                     key === 'CRYPTO' ? 'Always open — 24/7' : '';
    return `<div class="exchange-card ${selected}" data-key="${escapeHtml(key)}">
      <div class="exchange-flag">${flag}</div>
      <div class="exchange-name">${escapeHtml(ex.name)}</div>
      <div class="exchange-country">${escapeHtml(ex.country)} &middot; ${escapeHtml(ex.currency)}</div>
      <div class="exchange-desc">${escapeHtml(ex.description)}</div>
      <span class="market-status-badge ${key === 'CRYPTO' ? 'crypto' : st}">${stLabel}${localTime ? ' · ' + localTime : ''}</span>
      ${timeHint ? `<div class="exchange-hours">${escapeHtml(timeHint)}</div>` : ''}
      <div class="exchange-hours">Hours: ${escapeHtml(ex.open || '')} – ${escapeHtml(ex.close || '')} local</div>
      <div class="exchange-asset-count">${assetCount} assets tracked</div>
    </div>`;
  }).join('');

  grid.querySelectorAll('.exchange-card').forEach(card => {
    card.addEventListener('click', () => {
      const key = card.dataset.key;
      if (_pendingExchanges.has(key)) {
        if (_pendingExchanges.size <= 1) { toast('Required', 'At least one exchange must be selected.', 'info', 2000); return; }
        _pendingExchanges.delete(key);
        card.classList.remove('selected');
      } else {
        _pendingExchanges.add(key);
        card.classList.add('selected');
      }
      const infoEl2 = document.getElementById('exchange-selected-info');
      if (infoEl2) infoEl2.textContent = `${_pendingExchanges.size} exchange${_pendingExchanges.size !== 1 ? 's' : ''} selected`;
    });
  });
}

async function saveExchanges() {
  const list = [...(_pendingExchanges || new Set(['US']))];
  const btn = document.getElementById('exchange-save');
  setButtonLoading(btn, true, 'Applying…');
  try {
    await api('/api/preferences', { method: 'PUT', body: JSON.stringify({ preferred_exchanges: list }) });
    _saveExchangesToStorage(list);
    document.getElementById('exchange-modal').classList.add('hidden');
    _pendingExchanges = null;
    state.activeExchangeFilter = null;
    renderMarketStatusStrip();
    renderExchangeFilterPills();
    toast('Markets updated', `Tracking ${list.length} exchange${list.length !== 1 ? 's' : ''}. Refreshing signals…`, 'success');
    loadDashboard();
  } catch (err) {
    toast('Failed', err.message, 'error');
  } finally { setButtonLoading(btn, false); }
}

function closeExchangeModal() {
  document.getElementById('exchange-modal').classList.add('hidden');
  _pendingExchanges = null;
}

document.getElementById('exchange-modal-close').addEventListener('click', closeExchangeModal);
document.getElementById('exchange-save').addEventListener('click', saveExchanges);
document.getElementById('exchange-select-all').addEventListener('click', () => {
  Object.keys(_exchangeData).forEach(k => _pendingExchanges.add(k));
  _renderExchangeGrid();
});
document.getElementById('exchange-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('exchange-modal')) closeExchangeModal();
});
document.getElementById('manage-exchanges-btn').addEventListener('click', () => {
  if (!Object.keys(_exchangeData).length) {
    loadExchanges().then(openExchangeModal);
  } else {
    openExchangeModal();
  }
});

// ─── Smart Dashboard Search ─────────────────────────────────────────────────
// Tier 1: instant DOM filter (already-rendered cards) — zero latency
// Tier 2: autocomplete from catalogue (/api/signals/search) — pure in-memory
// Tier 3: on-demand live fetch (/api/signals/asset/{ticker}) — Yahoo Finance
// ─────────────────────────────────────────────────────────────────────────────
const _searchInput    = document.getElementById('dashboard-asset-search');
const _searchDropdown = document.getElementById('asset-search-dropdown');
const _searchDot      = document.getElementById('search-loading-dot');
let _searchDebounce   = null;
let _searchFetchTimer = null;
let _dropdownFocusIdx = -1;
let _dropdownItems    = [];

function _closeDropdown() {
  _searchDropdown.classList.add('hidden');
  _dropdownFocusIdx = -1;
  _dropdownItems = [];
}

function _positionDropdown() {
  const rect = _searchInput.getBoundingClientRect();
  _searchDropdown.style.top  = (rect.bottom + 6) + 'px';
  _searchDropdown.style.left = rect.left + 'px';
  _searchDropdown.style.width = rect.width + 'px';
}

function _renderDropdownItems(items, liveSignal) {
  if (!items.length && !liveSignal) { _closeDropdown(); return; }
  _positionDropdown();
  _searchDropdown.classList.remove('hidden');
  _dropdownItems = items;

  const exchFlags = { US:'🇺🇸',NSE:'🇮🇳',LSE:'🇬🇧',XETRA:'🇩🇪',TSE:'🇯🇵',HKEX:'🇭🇰',ASX:'🇦🇺',TSX:'🇨🇦',CRYPTO:'₿' };
  let html = '';

  if (liveSignal) {
    const sig = liveSignal;
    const rawLivePrice = Number(sig.last_price);
    const price = rawLivePrice.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: rawLivePrice < 1 ? 6 : 2 });
    const ex = sig.exchange || 'US';
    html += `<div class="asd-group-label">Live result — fetched from Yahoo Finance</div>
      <div class="asd-item" data-ticker="${escapeHtml(sig.asset)}" data-live="1">
        <span class="asd-ticker">${escapeHtml(sig.asset)}</span>
        <span class="asd-exch">${exchFlags[ex] || ''} ${escapeHtml(ex)}</span>
        <span class="asd-price">${price}</span>
        <span class="asd-signal ${escapeHtml(sig.signal_type)}">${escapeHtml(sig.signal_type)}</span>
      </div>`;
  }

  if (items.length) {
    html += `<div class="asd-group-label">Universe catalogue — ${items.length} match${items.length !== 1 ? 'es' : ''}</div>`;
    items.forEach((item) => {
      const flag = exchFlags[item.exchange] || '🌐';
      html += `<div class="asd-item" data-ticker="${escapeHtml(item.ticker)}" role="option">
        <span class="asd-ticker">${escapeHtml(item.ticker)}</span>
        <span class="asd-exch">${flag} ${escapeHtml(item.exchange)}</span>
      </div>`;
    });
  }
  _searchDropdown.innerHTML = html;
  _searchDropdown.querySelectorAll('.asd-item').forEach(el => {
    el.addEventListener('mousedown', (e) => { e.preventDefault(); _selectTicker(el.dataset.ticker, el.dataset.live === '1'); });
  });
}

async function _selectTicker(ticker, isLive) {
  _searchInput.value = ticker;
  state.dashSearchQuery = ticker;
  _closeDropdown();
  if (_searchDot) _searchDot.style.display = 'none';

  // Instantly filter existing cards
  _applyDashboardFilter();

  // Check if it's already in the grid
  const existing = document.getElementById('signal-' + ticker);
  if (existing) {
    existing.scrollIntoView({ behavior: 'smooth', block: 'center' });
    existing.classList.add('flash-update');
    setTimeout(() => existing.classList.remove('flash-update'), 1200);
    return;
  }

  // Fetch live signal and inject a card at top of grid
  if (_searchDot) _searchDot.style.display = '';
  try {
    const sig = await api(`/api/signals/asset/${encodeURIComponent(ticker)}`, {}, { silent: true });
    if (sig) _injectOnDemandCard(sig);
  } catch (e) {
    toast('Not found', `No market data for "${ticker}". Check the symbol format.`, 'error', 3000);
  } finally {
    if (_searchDot) _searchDot.style.display = 'none';
  }
}

function _injectOnDemandCard(sig) {
  const grid = document.getElementById('signal-grid');
  if (!grid) return;
  const rawAsset  = String(sig.asset);
  const asset     = escapeHtml(rawAsset);
  const sigType   = escapeHtml(String(sig.signal_type));
  const rsi       = sig.features?.rsi != null ? Number(sig.features.rsi).toFixed(1) : 'N/A';
  const mom       = sig.features?.momentum != null ? (Number(sig.features.momentum) * 100).toFixed(1) : 'N/A';
  const macd      = sig.features?.macd_histogram != null ? Number(sig.features.macd_histogram).toFixed(3) : 'N/A';
  const bbw       = sig.features?.bb_width != null ? Number(sig.features.bb_width).toFixed(3) : 'N/A';
  const confPct   = Math.min(100, Math.max(0, Math.round(Number(sig.confidence) * 100)));
  const rawPrice  = Number(sig.last_price);
  const price     = rawPrice.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: rawPrice < 1 ? 6 : 2 });
  const exchCodes = { US:'[US]',NSE:'[NSE]',LSE:'[LSE]',XETRA:'[XETRA]',TSE:'[TSE]',HKEX:'[HKEX]',ASX:'[ASX]',TSX:'[TSX]',CRYPTO:'[CRYPTO]' };
  const flag      = exchCodes[sig.exchange || 'US'] || '[INTL]';
  const companyName = escapeHtml(sig.company_name || rawAsset);
  const sector    = escapeHtml(sig.sector || '');
  const insight   = escapeHtml(sig.insight || '');
  const changePct = sig.change_pct != null ? Number(sig.change_pct) : null;
  const changeStr = changePct != null ? `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%` : '';
  const changeClass = changePct != null ? (changePct >= 0 ? 'positive' : 'negative') : '';

  // 3-month price range bar (renamed from 52w in signal engine fix B4)
  let rangeBarHtml = '';
  // FIX F8: fields were renamed high_3mo/low_3mo in signal_engine.py (B4).
  // The old high_52w/low_52w keys no longer exist on on-demand signal cards.
  const loP = sig.low_3mo ?? sig.low_52w;    // support both names during rollout
  const hiP = sig.high_3mo ?? sig.high_52w;
  if (hiP && loP && hiP > loP) {
    const pct = Math.min(100, Math.max(0, ((rawPrice - loP) / (hiP - loP)) * 100));
    const lo  = rawPrice < 1 ? loP.toFixed(4) : loP.toFixed(2);
    const hi  = rawPrice < 1 ? hiP.toFixed(4) : hiP.toFixed(2);
    rangeBarHtml = `
      <div class="sig-range-wrap">
        <span class="sig-range-label">3Mo</span>
        <div class="sig-range-bar"><div class="sig-range-fill" style="left:${pct.toFixed(1)}%"></div></div>
        <span class="sig-range-lo">$${lo}</span>–<span class="sig-range-hi">$${hi}</span>
      </div>`;
  }

  const old = document.getElementById(`signal-${rawAsset}`);
  if (old) old.remove();

  const div = document.createElement('div');
  div.className = `signal-card sig-${sigType} on-demand-card flash-update`;
  div.id = `signal-${rawAsset}`;
  div.innerHTML = `
    <div class="sig-header">
      <div>
        <div class="sig-company-name">${companyName}</div>
        <div class="asset">${asset} <span class="sig-exch-code">${flag}</span>${sector ? ` <span class="sig-sector-pill">${sector}</span>` : ''}</div>
        <div class="price">${price} ${changePct != null ? `<span class="sig-change ${changeClass}">${changeStr}</span>` : ''}</div>
      </div>
      <div style="display:flex;align-items:flex-start;gap:8px">
        <span class="badge ${sigType}">${sigType}</span>
        <button class="bookmark-btn" data-ticker="${asset}" title="Add ${asset} to watchlist">☆</button>
      </div>
    </div>
    <div class="confidence-bar"><div class="confidence-fill" data-target="${confPct}"></div></div>
    <div class="confidence-label"><span>Confidence</span><span><b>${confPct}%</b></span></div>
    ${rangeBarHtml}
    ${insight ? `<div class="sig-insight">Insight: ${insight}</div>` : ''}
    <div class="features-row">
      <span title="Relative Strength Index">RSI <b>${rsi}</b></span>
      <span title="20-day Momentum">Mom <b>${mom}%</b></span>
      <span title="MACD Histogram">MACD <b>${macd}</b></span>
      <span title="Bollinger Width">BBW <b>${bbw}</b></span>
    </div>
    <div class="sig-live-badge">LIVE · Yahoo Finance</div>
  `;
  grid.prepend(div);
  requestAnimationFrame(() => { div.querySelector('.confidence-fill').style.width = confPct + '%'; });
  div.querySelector('.bookmark-btn').addEventListener('click', () => addToWatchlist(sig.asset));
  div.scrollIntoView({ behavior: 'smooth', block: 'start' });
  toast('Signal fetched', `Live data loaded for ${sig.asset} — ${sigType} @ ${price}`, 'success', 3500);
}

// Debounced search: instant DOM filter + autocomplete + live fetch
_searchInput.addEventListener('input', (e) => {
  const q = e.target.value.trim();
  state.dashSearchQuery = q;

  // Tier 1: instant DOM filter (zero latency)
  _applyDashboardFilter();

  clearTimeout(_searchDebounce);
  clearTimeout(_searchFetchTimer);

  if (!q) { _closeDropdown(); return; }

  // Tier 2: catalogue autocomplete (250ms debounce)
  _searchDebounce = setTimeout(async () => {
    try {
      const data = await api(`/api/signals/search?q=${encodeURIComponent(q)}`, {}, { silent: true });
      _renderDropdownItems(data.results || [], null);
    } catch (_) { _closeDropdown(); }
  }, 250);

  // Tier 3: live fetch if no cards match (600ms debounce, only if ≥2 chars)
  if (q.length >= 2) {
    _searchFetchTimer = setTimeout(async () => {
      const existing = document.getElementById('signal-' + q.toUpperCase());
      if (existing) return;  // already in grid
      if (_searchDot) _searchDot.style.display = '';
      try {
        const sig = await api(`/api/signals/asset/${encodeURIComponent(q)}`, {}, { silent: true });
        if (sig && sig.asset) _renderDropdownItems(_dropdownItems, sig);
      } catch (_) { /* ticker not found — silently ignore */ }
      finally { if (_searchDot) _searchDot.style.display = 'none'; }
    }, 600);
  }
});

_searchInput.addEventListener('keydown', (e) => {
  const items = _searchDropdown.querySelectorAll('.asd-item');
  if (e.key === 'Escape') { _closeDropdown(); _searchInput.blur(); return; }
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    // Clamp to last item; start from -1 so first Down selects item 0
    _dropdownFocusIdx = Math.min(_dropdownFocusIdx + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle('focused', i === _dropdownFocusIdx));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    // Allow going back to -1 (nothing focused = back to free-text input)
    _dropdownFocusIdx = Math.max(_dropdownFocusIdx - 1, -1);
    items.forEach((el, i) => el.classList.toggle('focused', i === _dropdownFocusIdx));
  } else if (e.key === 'Enter') {
    e.preventDefault();
    const focused = _searchDropdown.querySelector('.asd-item.focused');
    if (focused) _selectTicker(focused.dataset.ticker, focused.dataset.live === '1');
    else if (_searchInput.value.trim()) _selectTicker(_searchInput.value.trim().toUpperCase(), false);
  }
});

document.addEventListener('click', (e) => {
  if (!_searchDropdown.contains(e.target) && e.target !== _searchInput) _closeDropdown();
});
_searchInput.addEventListener('focus', () => {
  if (_searchInput.value.trim() && _dropdownItems.length) {
    _positionDropdown();
    _searchDropdown.classList.remove('hidden');
  }
});


// Refresh market status strip every 60s
setInterval(() => {
  if (state.token && state.activeView === 'dashboard') loadExchanges();
}, 60000);

// ===========================================================================
// Watchlist Manager
// ===========================================================================
const DEFAULT_WATCHLIST = [

  'AAPL','MSFT','NVDA','GOOGL','META',
  'TSLA','AMZN','JPM','V','JNJ',
  'XOM','SPY','QQQ','GLD','COIN',
  'NFLX','AMD','BKNG','LLY','TSM',
];

let _pendingWatchlist = null;

function _saveWatchlistToStorage(list) {
  localStorage.setItem('qs_watchlist', JSON.stringify(list));
  state.watchlist = new Set(list);
}

function _currentWatchlist() {
  return state.watchlist.size ? [...state.watchlist] : [...DEFAULT_WATCHLIST];
}

function openWatchlistModal() {
  _pendingWatchlist = new Set(_currentWatchlist());
  _renderWatchlistModal();
  document.getElementById('watchlist-modal').classList.remove('hidden');
  document.getElementById('watchlist-search').focus();
}

function _renderWatchlistModal(filter = '') {
  const container = document.getElementById('watchlist-asset-list');
  const filterLow = filter.toLowerCase().trim();
  let html = '';
  let total = 0;
  const tracked = new Set(state.meta?.tracked_assets || []);

  for (const grp of ASSET_GROUPS) {
    const matchesGroup = !filterLow || grp.label.toLowerCase().includes(filterLow);
    const items = grp.prefix.filter(t =>
      tracked.has(t) && (!filterLow || matchesGroup || t.toLowerCase().includes(filterLow))
    );
    if (!items.length) continue;
    html += `<div class="watchlist-group-header">${escapeHtml(grp.label)}</div>`;
    for (const ticker of items) {
      const active = _pendingWatchlist.has(ticker) ? 'active' : '';
      html += `<div class="watchlist-asset-row ${active}" data-ticker="${escapeHtml(ticker)}">
        <span class="watchlist-ticker">${escapeHtml(ticker)}</span>
        <span class="watchlist-toggle" title="${active ? 'Remove' : 'Add'} ${escapeHtml(ticker)}"></span>
      </div>`;
    }
    total += items.length;
  }
  if (!total) {
    html = `<div class="empty-state" style="padding:40px 0;">No assets match "${escapeHtml(filter)}"</div>`;
  }
  container.innerHTML = html;
  _updateWatchlistCount();
  container.querySelectorAll('.watchlist-asset-row').forEach(row => {
    row.addEventListener('click', () => {
      const ticker = row.dataset.ticker;
      if (_pendingWatchlist.has(ticker)) {
        if (_pendingWatchlist.size <= 1) { toast('Required', 'Watchlist must have at least 1 asset.', 'info', 2000); return; }
        _pendingWatchlist.delete(ticker);
        row.classList.remove('active');
      } else {
        if (_pendingWatchlist.size >= 50) { toast('Limit reached', 'Watchlist is capped at 50 assets.', 'info', 2500); return; }
        _pendingWatchlist.add(ticker);
        row.classList.add('active');
      }
      _updateWatchlistCount();
    });
  });
}

function _updateWatchlistCount() {
  const el = document.getElementById('watchlist-selected-count');
  if (el) el.textContent = `${_pendingWatchlist?.size || 0} / 50 selected`;
}

async function saveWatchlist() {
  const list = [...(_pendingWatchlist || new Set(DEFAULT_WATCHLIST))];
  const btn = document.getElementById('watchlist-save');
  setButtonLoading(btn, true, 'Saving…');
  try {
    await api('/api/watchlist', { method: 'PUT', body: JSON.stringify({ watchlist: list }) });
    _saveWatchlistToStorage(list);
    document.getElementById('watchlist-modal').classList.add('hidden');
    _pendingWatchlist = null;
    toast('Watchlist updated', `Now tracking ${list.length} assets on your dashboard.`, 'success');
    loadDashboard();
  } catch (err) {
    toast('Save failed', err.message, 'error');
  } finally { setButtonLoading(btn, false); }
}

function closeWatchlistModal() {
  document.getElementById('watchlist-modal').classList.add('hidden');
  _pendingWatchlist = null;
}

document.getElementById('watchlist-modal-close').addEventListener('click', closeWatchlistModal);
document.getElementById('watchlist-save').addEventListener('click', saveWatchlist);
document.getElementById('watchlist-reset').addEventListener('click', () => {
  _pendingWatchlist = new Set(DEFAULT_WATCHLIST);
  document.getElementById('watchlist-search').value = '';
  _renderWatchlistModal();
});
document.getElementById('watchlist-modal').addEventListener('click', (e) => {
  if (e.target === document.getElementById('watchlist-modal')) closeWatchlistModal();
});
document.getElementById('watchlist-search').addEventListener('input', (e) => {
  _renderWatchlistModal(e.target.value);
});
document.getElementById('manage-watchlist-btn').addEventListener('click', openWatchlistModal);

async function removeFromWatchlist(ticker) {
  const current = _currentWatchlist().filter(t => t !== ticker);
  if (!current.length) { toast('Cannot remove', 'Watchlist must have at least one asset.', 'info'); return; }
  try {
    await api(`/api/watchlist/${encodeURIComponent(ticker)}`, { method: 'DELETE' }, { silent: true });
    _saveWatchlistToStorage(current);
    loadDashboard(true);
    toast('Removed', `${ticker} removed from watchlist.`, 'info', 2200);
  } catch (err) { toast('Failed', err.message, 'error'); }
}

async function addToWatchlist(ticker) {
  const current = _currentWatchlist();
  if (current.includes(ticker)) { toast('Already tracked', `${ticker} is already in your watchlist.`, 'info', 2000); return; }
  if (current.length >= 50) { toast('Limit reached', 'Watchlist is limited to 50 assets. Remove one first.', 'info'); return; }
  try {
    await api(`/api/watchlist/${encodeURIComponent(ticker)}`, { method: 'POST' }, { silent: true });
    _saveWatchlistToStorage([...current, ticker]);
    toast('Added', `${ticker} added to your watchlist!`, 'success', 2500);
    // Update the bookmark button on the injected card
    const card = document.getElementById(`signal-${ticker}`);
    if (card) {
      const btn = card.querySelector('.bookmark-btn');
      if (btn) { btn.textContent = '★'; btn.classList.add('bookmarked'); btn.onclick = () => removeFromWatchlist(ticker); }
    }
  } catch (err) { toast('Failed', err.message, 'error'); }
}

// ─── 3D Background Initialisation ─────────────────────────────────────────────
// Auth canvas starts immediately (vivid mode); app canvas starts on login
window.addEventListener('load', () => {
  const authCanvas = document.getElementById('auth-bg-canvas');
  if (authCanvas) {
    authCanvas.classList.add('vivid');
    // Wait for Three.js to be available (loaded deferred)
    const tryInit = () => {
      if (window.THREE) {
        QS3D.init('auth-bg-canvas', 'auth');
      } else {
        setTimeout(tryInit, 100);
      }
    };
    tryInit();
  }
  // If already logged in (token persisted), start app canvas immediately
  if (state.token) {
    const appCanvas = document.getElementById('app-bg-canvas');
    if (appCanvas) {
      authCanvas?.style && (authCanvas.style.display = 'none');
      appCanvas.style.display = '';
      const tryInit2 = () => {
        if (window.THREE) QS3D.init('app-bg-canvas', 'dashboard');
        else setTimeout(tryInit2, 100);
      };
      tryInit2();
    }
  }
});

function connectSignalStream() {

  if (!state.token || state.signalSocket) return;
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  // URL-encode the JWT to prevent header parse failures on tokens with
  // special characters ('+', '/', '=') that are not valid in WS subprotocol values.
  // The server decodes it identically since it only calls decode_access_token().
  const safeToken = encodeURIComponent(state.token);
  const socket = new WebSocket(`${scheme}//${location.host}/api/signals/stream`, ['qs', safeToken]);
  state.signalSocket = socket;
  setLiveIndicator('reconnecting');
  socket.onopen = () => { state.signalReconnectMs = 1000; setLiveIndicator('connected'); };
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      // Persist to cache for instant next-load
      try { localStorage.setItem('qs_signal_cache', JSON.stringify(data)); } catch {}
      if (state.activeView === 'dashboard') loadDashboard(true, data);
    } catch (err) {
      console.error('WebSocket message handling error:', err);
    }
  };
  socket.onerror = (err) => {
    console.warn('WebSocket error:', err);
  };
  socket.onclose = () => {
    if (state.signalSocket !== socket) return;
    state.signalSocket = null;
    const delay = state.signalReconnectMs;
    state.signalReconnectMs = Math.min(30000, delay * 2);
    if (state.token) {
      setLiveIndicator('reconnecting');
      setTimeout(connectSignalStream, delay);
    } else {
      setLiveIndicator('disconnected');
    }
  };
}

async function loadStrategies() {
  const list = document.getElementById('strategy-list');
  const strategies = await api('/api/strategies', {}, { silent: true }).catch(() => []);
  list.innerHTML = strategies.map((s) => `<div class="order-row"><span>${escapeHtml(s.name)} · ${escapeHtml((s.assets || []).join(', '))}</span><span>${Number(s.config.fast_window)}/${Number(s.config.slow_window)} day</span></div>`).join('') || '<div class="empty-state">No saved strategies yet.</div>';
}

document.getElementById('strategy-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const error = document.getElementById('strategy-error'); error.textContent = '';
  const btn = e.currentTarget.querySelector('button[type=submit]');
  const fastW = Number(document.getElementById('strategy-fast').value);
  const slowW = Number(document.getElementById('strategy-slow').value);
  // M8 FIX: pre-validate window relationship on the frontend to give a clear
  // error message instead of a raw 400 from the server.
  if (fastW >= slowW) {
    error.textContent = `Fast window (${fastW}) must be smaller than slow window (${slowW}).`;
    return;
  }
  const body = {
    name: document.getElementById('strategy-name').value,
    asset: document.getElementById('strategy-asset').value,
    fast_window: fastW,
    slow_window: slowW,
    period: document.getElementById('strategy-period').value,
  };
  setButtonLoading(btn, true, 'Backtesting historical data…');
  try {
    await api('/api/strategies', { method: 'POST', body: JSON.stringify(body) });
    const result = await api('/api/backtests', { method: 'POST', body: JSON.stringify(body) });
    document.getElementById('backtest-result').innerHTML = `<div class="metrics-row"><div class="metric-box"><div class="val">${(result.total_return * 100).toFixed(1)}%</div><div class="lbl">Total Return</div></div><div class="metric-box"><div class="val">${result.sharpe_ratio.toFixed(2)}</div><div class="lbl">Sharpe</div></div><div class="metric-box"><div class="val">${(result.max_drawdown * 100).toFixed(1)}%</div><div class="lbl">Max Drawdown</div></div></div><p class="hint">${result.asset} · ${result.period} · ${result.total_trades} order events · historical simulation only.</p>`;
    await loadStrategies(); toast('Backtest completed', `${result.asset} total return: ${(result.total_return * 100).toFixed(1)}%`, 'success');
  } catch (err) { error.textContent = err.message; } finally { setButtonLoading(btn, false); }
});

const onboardingSteps = [
  ['Welcome to QuantumSentinel', 'This five-step tour shows how to read signals, test a strategy, place a paper order, review risk, and check cryptographic health.'],
  ['1 · Read signals', 'The dashboard combines market indicators with a quantum-inspired optimizer. Confidence describes model conviction, not certainty.'],
  ['2 · Test before trading', 'Use Strategies to tune a moving-average template and validate it against historical data.'],
  ['3 · Paper trade only', 'Orders are simulated or routed only to an Alpaca paper account. No real-money brokerage is included.'],
  ['4 · Monitor safety', 'Portfolio shows risk metrics and Security shows key freshness plus verifiable audit records.'],
];
let onboardingIndex = 0;
function startOnboardingIfNeeded() {
  if (!state.beginner || localStorage.getItem('qs_onboarding_complete') === 'true') return;
  const modal = document.getElementById('onboarding-modal'); modal.classList.remove('hidden'); renderOnboarding();
}
function renderOnboarding() {
  document.getElementById('onboarding-title').textContent = onboardingSteps[onboardingIndex][0];
  document.getElementById('onboarding-body').textContent = onboardingSteps[onboardingIndex][1];
  document.querySelector('#onboarding-next .btn-label').textContent = onboardingIndex === onboardingSteps.length - 1 ? 'Finish tour' : 'Next';
}
document.getElementById('onboarding-next').addEventListener('click', () => { onboardingIndex++; if (onboardingIndex >= onboardingSteps.length) { localStorage.setItem('qs_onboarding_complete', 'true'); document.getElementById('onboarding-modal').classList.add('hidden'); } else renderOnboarding(); });
document.getElementById('onboarding-skip').addEventListener('click', () => { localStorage.setItem('qs_onboarding_complete', 'true'); document.getElementById('onboarding-modal').classList.add('hidden'); });

function skeletonGrid(container, count, cardClass) {
  container.innerHTML = Array.from({ length: count })
    .map(() => `<div class="skeleton ${cardClass}"></div>`).join('');
}

// ===========================================================================
// Dashboard — real-time auto-refresh countdown
// ===========================================================================
let _signalAgeTimer = null;
let _refreshCountdown = null;
const SIGNAL_REFRESH_INTERVAL = 30; // seconds, matches backend CACHE_TTL_SECONDS

function _startSignalAgeClock(generatedAt) {
  if (_signalAgeTimer) clearInterval(_signalAgeTimer);
  const ageEl = document.getElementById('signal-age-live');
  if (!ageEl) return;
  function update() {
    const secs = Math.round((Date.now() / 1000) - generatedAt);
    if (secs < 60) ageEl.textContent = `(${secs}s ago)`;
    else ageEl.textContent = `(${Math.floor(secs / 60)}m ${secs % 60}s ago)`;
  }
  update();
  _signalAgeTimer = setInterval(update, 1000);
}

function startRefreshCountdown() {
  if (_refreshCountdown) clearInterval(_refreshCountdown);
  let remaining = SIGNAL_REFRESH_INTERVAL;
  const countdownEl = document.getElementById('refresh-countdown');
  function tick() {
    if (!countdownEl) return;
    // Decrement first so display is always accurate (avoids showing 0 twice)
    remaining--;
    if (remaining <= 0) {
      remaining = SIGNAL_REFRESH_INTERVAL;
      countdownEl.textContent = 'Refreshing…';
      if (state.activeView === 'dashboard' && state.token) {
        loadDashboard(true);
      }
    } else {
      countdownEl.textContent = `Next refresh in ${remaining}s`;
    }
  }
  // Show initial state immediately before first interval fires
  if (countdownEl) countdownEl.textContent = `Next refresh in ${remaining}s`;
  _refreshCountdown = setInterval(tick, 1000);
}

async function loadDashboard(isPoll, streamedSignals = null) {
  const grid = document.getElementById('signal-grid');

  // INSTANT RENDER: show cached signals immediately on first load (zero latency)
  if (!isPoll && !streamedSignals && state._signalCache) {
    _renderSignalGrid(grid, state._signalCache, true /* fromCache */);
  } else if (!isPoll && !grid.children.length) {
    skeletonGrid(grid, 8, 'skeleton-card');
  }

  const signals = streamedSignals || await api('/api/signals/latest', {}, { silent: isPoll }).catch(() => null);
  if (!signals) return;

  // Persist to cache
  state._signalCache = signals;
  try { localStorage.setItem('qs_signal_cache', JSON.stringify(signals)); } catch {}

  // Update watchlist state from server response
  if (signals.watchlist?.length) _saveWatchlistToStorage(signals.watchlist);

  const genTime = new Date(signals.generated_at * 1000).toLocaleTimeString();
  // total_assets = full preloaded count; n_assets = currently displayed (filtered to watchlist)
  const totalAssets = signals.total_assets ?? signals.n_assets;
  document.getElementById('signal-meta').innerHTML =
    `Engine: ${signals.pipeline_ms} ms &middot; SBA: ${signals.sba_ms} ms &middot; ` +
    `<span class="signal-count-chip">${signals.n_assets} in view &middot; ${totalAssets} generated</span> ` +
    `&middot; ${escapeHtml(genTime)} <span class="signal-age" id="signal-age-live"></span>` +
    (signals.error ? ` &middot; <span style="color:var(--red)">&#9888; ${escapeHtml(signals.error)}</span>` : '');
  _startSignalAgeClock(signals.generated_at);
  cacheSignalPrices(signals);
  if (state.activeView === 'trading') updateOrderPricePreview();

  _renderSignalGrid(grid, signals, false);

  if (!isPoll) {
    api('/api/security/health', {}, { silent: true }).then((health) => {
      if (health) animateScoreRing(health.quantum_safety_score);
    }).catch(() => {});
  }
}

function _renderSignalGrid(grid, signals, fromCache) {
  const watched = _currentWatchlist();
  const sigList = signals.signals || [];

  if (!sigList.length && !fromCache) {
    // Show empty watchlist state with CTA
    grid.innerHTML = `<div class="watchlist-empty">
      <h3>Your watchlist is empty</h3>
      <p>Add assets to your watchlist to see live signals. Start with our curated blue-chip selection.</p>
      <button class="btn-primary ripple-btn" onclick="openWatchlistModal()" style="display:inline-block;width:auto;padding:10px 24px;">&#9733; Manage Watchlist</button>
    </div>`;
    return;
  }

  grid.innerHTML = sigList.map((s, i) => {
    const asset     = escapeHtml(String(s.asset));
    const sigType   = escapeHtml(String(s.signal_type));
    const rsi       = s.features?.rsi != null ? Number(s.features.rsi).toFixed(1) : 'N/A';
    const mom       = s.features?.momentum != null ? (Number(s.features.momentum) * 100).toFixed(1) : 'N/A';
    const macd      = s.features?.macd_histogram != null ? Number(s.features.macd_histogram).toFixed(3) : 'N/A';
    const bbw       = s.features?.bb_width != null ? Number(s.features.bb_width).toFixed(3) : 'N/A';
    const confPct   = Math.min(100, Math.max(0, Math.round(Number(s.confidence) * 100)));
    const rawP      = Number(s.last_price);
    const price     = rawP.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: rawP < 1 ? 6 : 2 });
    const exchCodes = { US:'[US]',NSE:'[NSE]',LSE:'[LSE]',XETRA:'[XETRA]',TSE:'[TSE]',HKEX:'[HKEX]',ASX:'[ASX]',TSX:'[TSX]',CRYPTO:'[CRYPTO]' };
    const flag      = exchCodes[s.exchange || 'US'] || '[INTL]';
    const companyName = escapeHtml(s.company_name || String(s.asset));
    const sector    = escapeHtml(s.sector || '');
    const insight   = escapeHtml(s.insight || '');
    return `
    <div class="signal-card sig-${sigType}" id="signal-${asset}" style="animation-delay:${i * 40}ms">
      <div class="sig-header">
        <div>
          <div class="sig-company-name">${companyName}</div>
          <div class="asset">${asset} <span class="sig-exch-code">${flag}</span>${sector ? ` <span class="sig-sector-pill">${sector}</span>` : ''}</div>
          <div class="price">${price}</div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:8px">
          <span class="badge ${sigType}">${sigType}</span>
          <button class="bookmark-btn bookmarked" data-ticker="${asset}" title="Remove ${asset} from watchlist">&#9733;</button>
        </div>
      </div>
      <div class="confidence-bar"><div class="confidence-fill" data-target="${confPct}"></div></div>
      <div class="confidence-label"><span>Confidence</span><span><b>${confPct}%</b></span></div>
      ${insight ? `<div class="sig-insight">Insight: ${insight}</div>` : ''}
      <div class="features-row">
        <span title="Relative Strength Index">RSI <b>${rsi}</b></span>
        <span title="20-day Momentum">Mom <b>${mom}%</b></span>
        <span title="MACD Histogram">MACD <b>${macd}</b></span>
        <span title="Bollinger Width">BBW <b>${bbw}</b></span>
      </div>
    </div>`;
  }).join('') || '<div class="empty-state">Signals loading — engine is warming up for your watchlist&hellip;</div>';

  // Animate confidence bars
  requestAnimationFrame(() => {
    grid.querySelectorAll('.confidence-fill').forEach((bar) => { bar.style.width = bar.dataset.target + '%'; });
  });

  // Flash changed signals
  sigList.forEach((s) => {
    const prev = state.lastSignals[s.asset];
    if (prev !== undefined && prev !== s.signal_type) {
      const card = document.getElementById(`signal-${s.asset}`);
      if (card) { card.classList.remove('flash-update'); requestAnimationFrame(() => card.classList.add('flash-update')); }
    }
    state.lastSignals[s.asset] = s.signal_type;
  });

  // Wire bookmark (remove) buttons
  grid.querySelectorAll('.bookmark-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeFromWatchlist(btn.dataset.ticker);
    });
  });

  // Re-apply active search / exchange filter so cards don't flash visible then disappear
  if (state.dashSearchQuery || state.activeExchangeFilter) {
    requestAnimationFrame(_applyDashboardFilter);
  }
}

document.getElementById('refresh-signals').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  const icon = document.getElementById('refresh-icon-el');
  // FIX F13: setButtonLoading replaces .btn-label innerHTML, destroying the .refresh-icon
  // span. Store the icon's presence BEFORE setButtonLoading mutates the DOM.
  const hadIcon = !!icon;
  if (icon) icon.classList.add('spinning');
  setButtonLoading(btn, true, ' Refreshing…');
  try {
    await api('/api/signals/refresh', { method: 'POST' });
    await loadDashboard();
    toast('Signals refreshed', 'SBA engine re-ran over live market data', 'success', 2200);
  } catch (err) {
    toast('Refresh failed', err.message, 'error');
  } finally {
    setButtonLoading(btn, false);
    // Re-query the icon AFTER restoring the button label
    const iconAfter = document.getElementById('refresh-icon-el');
    if (iconAfter) iconAfter.classList.remove('spinning');
  }
});

// ===========================================================================
// Trading
// ===========================================================================
function loadTrading() { refreshOrders(); updateOrderPricePreview(); updateAssetInfo(); _startOrderPricePolling(); }

// Live price preview -- always fetches fresh via /api/price
let _priceRefreshInterval = null;

function _startOrderPricePolling() {
  clearInterval(_priceRefreshInterval);
  _priceRefreshInterval = setInterval(() => {
    if (state.activeView === 'trading') updateOrderPricePreview();
  }, 10000);
}

async function updateOrderPricePreview() {
  const asset   = document.getElementById('order-asset')?.value;
  const preview = document.getElementById('order-price-preview');
  const valEl   = document.getElementById('order-price-val');
  const labelEl = document.getElementById('order-price-label');
  if (!preview || !valEl || !asset) return;
  try {
    const data = await api(`/api/price/${encodeURIComponent(asset)}`, {}, { silent: true });
    if (!data?.price) return;
    const rawP  = Number(data.price);
    const curr  = data.currency || 'USD';
    const price = rawP.toLocaleString('en-US', { style: 'currency', currency: 'USD',
                    maximumFractionDigits: rawP < 1 ? 6 : 2 });
    const chg   = data.change_pct != null ? Number(data.change_pct) : null;
    const chgHtml = chg != null && isFinite(chg)
      ? ` <span class="sig-change ${chg >= 0 ? 'positive' : 'negative'}">${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%</span>`
      : '';
    const age   = Math.round(Date.now() / 1000 - (data.fetched_at || 0));
    valEl.innerHTML = price + chgHtml;
    labelEl.textContent = `Live price (${curr})${age < 10 ? '' : ' · ' + age + 's ago'}`;
    preview.classList.remove('hidden');
    if (!state.lastSignalPrices) state.lastSignalPrices = {};
    state.lastSignalPrices[asset] = rawP;
  } catch (_) {
    const p2 = state.lastSignalPrices?.[asset];
    if (p2) {
      valEl.textContent = Number(p2).toLocaleString('en-US', { style: 'currency', currency: 'USD' });
      labelEl.textContent = 'Cached price (live fetch failed)';
      preview.classList.remove('hidden');
    }
  }
}

// Dynamic asset info -- updates order form based on instrument type
let _assetInfoEl = null;
async function updateAssetInfo() {
  const asset = document.getElementById('order-asset')?.value;
  if (!asset) return;
  if (!_assetInfoEl) {
    _assetInfoEl = document.createElement('div');
    _assetInfoEl.id = 'asset-info-bar';
    _assetInfoEl.className = 'asset-info-bar';
    const form = document.getElementById('order-form');
    if (form) form.insertBefore(_assetInfoEl, form.firstChild);
  }
  _assetInfoEl.innerHTML = '<span class="hint">Loading asset info...</span>';
  try {
    const info = await api(`/api/asset/info/${encodeURIComponent(asset)}`, {}, { silent: true });
    if (!info) { _assetInfoEl.innerHTML = ''; return; }
    const mktOpen  = info.market_open;
    const dotClass = mktOpen ? 'open' : 'closed';
    const dotText  = mktOpen ? 'Market Open' : 'Market Closed';
    const typeLabel = info.instrument_type || 'EQUITY';
    const fractLabel = info.fractional_allowed ? 'Fractional OK' : 'Whole shares only';
    const h24 = info.trading_24_7 ? ' · 24/7' : '';
    _assetInfoEl.innerHTML =
      `<span class="aib-name">${escapeHtml(info.company_name || asset)}</span>` +
      `<span class="aib-sep">·</span><span class="aib-type">${typeLabel}</span>` +
      `<span class="aib-sep">·</span><span class="aib-market"><span class="mkt-dot ${dotClass}"></span>${dotText}</span>` +
      `<span class="aib-sep">·</span><span class="aib-frac">${fractLabel}${h24}</span>`;
    // Warn if market closed + market order
    let warn = document.getElementById('market-closed-warn');
    if (!warn) {
      warn = document.createElement('div');
      warn.id = 'market-closed-warn';
      warn.className = 'hint warn';
      warn.style.cssText = 'color:var(--amber);margin-bottom:10px;';
      warn.textContent = 'Market is currently closed. Market orders will queue until next open.';
      const ot = document.getElementById('order-type');
      if (ot?.parentElement) ot.parentElement.after(warn);
    }
    const isMarketOrder = ['market','stop'].includes(document.getElementById('order-type')?.value);
    warn.style.display = (!mktOpen && isMarketOrder && !info.trading_24_7) ? '' : 'none';
  } catch (_) { _assetInfoEl.innerHTML = ''; }
}

// Store prices when signals load
function cacheSignalPrices(signals) {
  if (!state.lastSignalPrices) state.lastSignalPrices = {};
  (signals?.signals || []).forEach((s) => { state.lastSignalPrices[s.asset] = s.last_price; });
}


document.getElementById('order-type').addEventListener('change', (e) => {
  const type = e.target.value;
  document.getElementById('limit-price-wrap').classList.toggle('hidden', !['limit', 'stop_limit'].includes(type));
  document.getElementById('stop-price-wrap').classList.toggle('hidden', !['stop', 'stop_limit'].includes(type));
});

document.getElementById('order-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('order-error');
  const btn = document.getElementById('order-submit-btn');
  errEl.textContent = '';
  const orderType = document.getElementById('order-type').value;
  const body = {
    asset: document.getElementById('order-asset').value,
    side: document.getElementById('order-side').value,
    quantity: parseFloat(document.getElementById('order-qty').value) || 0,
    order_type: orderType,
    time_in_force: document.getElementById('order-tif').value,
    limit_price: ['limit', 'stop_limit'].includes(orderType)
      ? parseFloat(document.getElementById('order-limit-price').value) : null,
    stop_price: ['stop', 'stop_limit'].includes(orderType)
      ? parseFloat(document.getElementById('order-stop-price').value) : null,
  };
  // F1: validate numeric inputs before sending to server
  if (!body.asset) { errEl.textContent = 'Asset is required.'; return; }
  if (!isFinite(body.quantity) || body.quantity <= 0) {
    errEl.textContent = 'Please enter a valid positive quantity.'; return;
  }
  if (['limit','stop_limit'].includes(body.order_type) && (!isFinite(body.limit_price) || body.limit_price <= 0)) {
    errEl.textContent = 'Please enter a valid limit price > 0.'; return;
  }
  if (['stop','stop_limit'].includes(body.order_type) && (!isFinite(body.stop_price) || body.stop_price <= 0)) {
    errEl.textContent = 'Please enter a valid stop price > 0.'; return;
  }
  setButtonLoading(btn, true, 'Signing with ML-DSA-65…');
  try {
    const order = await api('/api/trading/orders', { method: 'POST', body: JSON.stringify(body) }, { silent: true });
    await refreshOrders();
    toast(
      order.status === 'FILLED' ? 'Order filled' : 'Order submitted',
      `${body.side.toUpperCase()} ${body.quantity} ${body.asset}${order.filled_price != null ? ' @ $' + Number(order.filled_price).toFixed(2) : ''}`,
      order.status === 'FILLED' ? 'success' : 'info'
    );
  } catch (err) { errEl.textContent = err.message; } finally { setButtonLoading(btn, false); }
});

async function cancelOrder(orderId, btn) {
  btn.disabled = true;
  btn.textContent = '…';
  try {
    await api(`/api/trading/orders/${orderId}`, { method: 'DELETE' }, { silent: true });
    await refreshOrders();
    toast('Order cancelled', 'The pending order has been cancelled.', 'info');
  } catch (err) {
    toast('Cancel failed', err.message, 'error');
    btn.disabled = false;
    btn.textContent = 'Cancel';
  }
}

// Raw order store for filtering
let _allOrders = [];

async function refreshOrders(isPoll) {
  const list = document.getElementById('order-list');
  if (!isPoll && !list.children.length) skeletonGrid(list, 4, 'skeleton-row');
  const orders = await api('/api/trading/orders', {}, { silent: isPoll });
  _allOrders = orders;
  renderOrderList(orders);
}

function filterOrderList() {
  const q = (document.getElementById('order-search')?.value || '').toLowerCase().trim();
  if (!q) { renderOrderList(_allOrders); return; }
  renderOrderList(_allOrders.filter((o) =>
    String(o.asset).toLowerCase().includes(q) ||
    String(o.status).toLowerCase().includes(q) ||
    String(o.side).toLowerCase().includes(q)
  ));
}

function renderOrderList(orders) {
  const list = document.getElementById('order-list');
  const cancellable = new Set(['PENDING', 'ACCEPTED', 'SUBMITTED']);
  list.innerHTML = orders.map((o, i) => {
    const side   = escapeHtml(String(o.side || '')).toUpperCase();
    const asset  = escapeHtml(String(o.asset));
    const status = escapeHtml(String(o.status));
    const orderId = escapeHtml(String(o.order_id));
    const qty    = Number(o.quantity);
    const qtyStr = qty < 1 ? qty.toFixed(6) : qty % 1 === 0 ? qty.toFixed(0) : qty.toFixed(4);
    const type   = escapeHtml(String(o.order_type || 'market'));
    const fillPrice = o.filled_price != null
      ? Number(o.filled_price).toLocaleString('en-US',{style:'currency',currency:'USD'})
      : '—'; // FIX F9: was `o.filled_price ?` which treats 0.0 as null (falsy)
    const cancelBtn = cancellable.has(o.status)
      ? `<button class="cancel-btn ripple-btn" data-order-id="${orderId}">Cancel</button>` : '';
    return `
    <div class="order-row" style="animation-delay:${i * 40}ms" data-order-id="${orderId}">
      <span class="asset-col">${asset}</span>
      <span class="side-col ${o.side?.toLowerCase()}">${side}</span>
      <span class="qty-col">${qty}</span>
      <span class="price-col">${fillPrice}</span>
      <span class="status status-${status}">${status}</span>
      ${cancelBtn}
    </div>`;
  }).join('') || '<div class="empty-state">No orders yet — place your first paper trade.</div>';

  list.querySelectorAll('.cancel-btn').forEach((btn) => {
    btn.addEventListener('click', () => cancelOrder(btn.dataset.orderId, btn));
  });
}

// ===========================================================================
// Portfolio
// ===========================================================================
async function loadPortfolio(isPoll) {
  const metricsEl = document.getElementById('risk-metrics');
  if (!isPoll && !metricsEl.children.length) skeletonGrid(metricsEl, 5, 'skeleton-card');

  // Use allSettled so one failing endpoint doesn't crash the entire portfolio view.
  const [posResult, metResult] = await Promise.allSettled([
    api('/api/portfolio/positions', {}, { silent: isPoll }),
    api('/api/portfolio/risk-metrics', {}, { silent: isPoll }),
  ]);
  const positions = posResult.status === 'fulfilled' ? posResult.value : [];
  const metrics   = metResult.status === 'fulfilled' ? metResult.value :
    { sharpe_ratio: 0, max_drawdown: 0, win_rate: 0, total_trades: 0, var_95: 0, var_99: 0, equity_curve: [] };
  if (posResult.status === 'rejected' && !isPoll) toast('Portfolio unavailable', 'Could not load positions — try again.', 'error', 4000);

  metricsEl.innerHTML = `
    <div class="metric-box"><div class="val" id="m-sharpe" data-raw-value="0">0</div><div class="lbl">Sharpe Ratio</div></div>
    <div class="metric-box"><div class="val" id="m-dd" data-raw-value="0">0%</div><div class="lbl">Max Drawdown</div></div>
    <div class="metric-box"><div class="val" id="m-win" data-raw-value="0">0%</div><div class="lbl">Win Rate</div></div>
    <div class="metric-box"><div class="val" id="m-trades" data-raw-value="0">0</div><div class="lbl">Filled Trades</div></div>
    <div class="metric-box"><div class="val" id="m-var" data-raw-value="0">0%</div><div class="lbl">VaR 95%</div></div>
    <div class="metric-box"><div class="val" id="m-var99" data-raw-value="0">0%</div><div class="lbl">VaR 99%</div></div>
  `;
  animateCounter(document.getElementById('m-sharpe'), metrics.sharpe_ratio, { decimals: 2 });
  animateCounter(document.getElementById('m-dd'), metrics.max_drawdown * 100, { decimals: 1, suffix: '%' });
  animateCounter(document.getElementById('m-win'), metrics.win_rate * 100, { decimals: 0, suffix: '%' });
  animateCounter(document.getElementById('m-trades'), metrics.total_trades, { decimals: 0 });
  animateCounter(document.getElementById('m-var'), metrics.var_95 * 100, { decimals: 2, suffix: '%' });
  animateCounter(document.getElementById('m-var99'), metrics.var_99 * 100, { decimals: 2, suffix: '%' });

  // Total portfolio value
  // Include realized_pnl so sold gains are not erased from the display total.
  // F3 FIX: sum unrealized from open positions + realized from ALL positions.
  // Realized PnL from fully-sold (qty=0) positions is NOT in the positions array
  // (server excludes zero-qty rows). We still show it if it was previously loaded
  // from risk_metrics (via equity_curve / trades). For open positions, realized
  // comes from partial sells tracked per-asset.
  const totalUnrealized = positions.reduce((s, p) => s + Number(p.unrealized_pnl || 0), 0);
  const totalRealized   = positions.reduce((s, p) => s + Number(p.realized_pnl  || 0), 0);
  const totalPnl = totalUnrealized + totalRealized;
  const totalEl = document.getElementById('portfolio-total');
  if (totalEl) {
    totalEl.textContent = totalPnl.toLocaleString('en-US', { style: 'currency', currency: 'USD', signDisplay: 'always' });
    totalEl.style.color = totalPnl >= 0 ? 'var(--green)' : 'var(--red)';
  }
  // Update topbar PnL chip
  const pnlChip = document.getElementById('topbar-pnl');
  if (pnlChip) {
    pnlChip.textContent = totalPnl.toLocaleString('en-US', { style: 'currency', currency: 'USD', signDisplay: 'always' });
    pnlChip.className = totalPnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    pnlChip.classList.remove('hidden');
  }

  const list = document.getElementById('positions-list');
  list.innerHTML = positions.map((p, i) => {
    const asset    = escapeHtml(String(p.asset));
    const qty      = Number(p.quantity);
    const qtyFmt   = qty < 1 ? qty.toFixed(6) : qty % 1 === 0 ? qty.toFixed(0) : qty.toFixed(4);
    const entry    = Number(p.avg_entry_price).toLocaleString('en-US', { style: 'currency', currency: 'USD' });
    const uPnl     = Number(p.unrealized_pnl);
    const rPnl     = Number(p.realized_pnl || 0);
    const uPnlFmt  = uPnl.toLocaleString('en-US', { style: 'currency', currency: 'USD', signDisplay: 'always' });
    const rPnlFmt  = rPnl.toLocaleString('en-US', { style: 'currency', currency: 'USD', signDisplay: 'always' });
    const uClass   = uPnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const rClass   = rPnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    return `
    <div class="pos-row" style="animation-delay:${i * 50}ms">
      <span style="font-weight:600">${asset}</span>
      <span>${qtyFmt} sh</span><!-- FIX F10: was raw qty, now uses qtyFmt for fractional display -->
      <span>${entry} avg</span>
      <span class="${uClass}" title="Unrealized P&amp;L">${uPnlFmt} <small style="opacity:.65;font-size:10px">unrl.</small></span>
      <span class="${rClass}" title="Realized P&amp;L">${rPnlFmt} <small style="opacity:.65;font-size:10px">rlzd.</small></span>
    </div>`;
  }).join('') || '<div class="empty-state">No open positions.</div>';

  animateEquityCurve(metrics.equity_curve || []);
}

document.getElementById('portfolio-export').addEventListener('click', async () => {
  const btn = document.getElementById('portfolio-export');
  setButtonLoading(btn, true, 'Generating…');
  try {
    const response = await fetch('/api/portfolio/export', {
      headers: { Authorization: 'Bearer ' + state.token },
    });
    if (!response.ok) throw new Error('Export could not be generated');
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url; link.download = 'quantumsentinel-portfolio.csv'; link.click();
    URL.revokeObjectURL(url);
  } catch (err) { toast('Export failed', err.message, 'error'); } finally { setButtonLoading(btn, false); }
});

let equityAnimFrame = null;
function animateEquityCurve(curve) {
  const canvas = document.getElementById('equity-canvas');
  const ctx = canvas.getContext('2d');
  if (equityAnimFrame) cancelAnimationFrame(equityAnimFrame);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!curve.length) {
    ctx.fillStyle = 'var(--text-3, #7A8599)'; ctx.font = '12px "IBM Plex Mono", monospace';
    ctx.fillText('No equity history yet — place a trade to begin tracking.', 20, 130);
    return;
  }
  const min = Math.min(...curve), max = Math.max(...curve);
  const pad = 24;
  const w = canvas.width - pad * 2, h = canvas.height - pad * 2;
  const points = curve.map((v, i) => ({
    x: pad + (i / (curve.length - 1 || 1)) * w,
    y: pad + h - ((v - min) / (max - min || 1)) * h,
  }));

  const duration = 900;
  const start = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const revealCount = Math.max(1, Math.floor(points.length * t));
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const visible = points.slice(0, revealCount);
    const last = visible[visible.length - 1];

    // Gradient fill — institutional blue on white canvas
    const gradient = ctx.createLinearGradient(0, pad, 0, pad + h);
    gradient.addColorStop(0, 'rgba(24,66,168,0.12)');
    gradient.addColorStop(1, 'rgba(24,66,168,0.01)');
    ctx.beginPath();
    visible.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.lineTo(last.x, pad + h);
    ctx.lineTo(visible[0].x, pad + h);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Grid lines (subtle)
    ctx.strokeStyle = 'rgba(216,220,230,0.6)';
    ctx.lineWidth = 0.5;
    for (let gi = 0; gi <= 4; gi++) {
      const gy = pad + (gi / 4) * h;
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(pad + w, gy); ctx.stroke();
    }

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#1842A8';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    visible.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.stroke();

    // Endpoint dot
    ctx.beginPath();
    ctx.fillStyle = '#1842A8';
    ctx.arc(last.x, last.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = '#FFFFFF';
    ctx.arc(last.x, last.y, 2, 0, Math.PI * 2);
    ctx.fill();

    if (t < 1) equityAnimFrame = requestAnimationFrame(frame);
  }
  equityAnimFrame = requestAnimationFrame(frame);
}

// ===========================================================================
// Security
// ===========================================================================
async function loadSecurity(isPoll) {
  const keyEl = document.getElementById('key-health');
  if (!isPoll && !keyEl.children.length) skeletonGrid(keyEl, 2, 'skeleton-row');

  const health = await api('/api/security/health', {}, { silent: isPoll });
  animateScoreRing(health.quantum_safety_score);

  // Security tab badge — count keys with status RED or YELLOW (aging)
  const agingKeys = (health.keys || []).filter((k) => k.status === 'RED' || k.rotation_due_in_days < 15);
  const badge = document.getElementById('security-badge');
  if (badge) badge.textContent = agingKeys.length > 0 ? String(agingKeys.length) : '';

  keyEl.innerHTML = health.keys.map((k, i) => {
    const status = escapeHtml(String(k.status));
    const algo   = escapeHtml(String(k.algorithm));
    const urgent = k.rotation_due_in_days < 15 ? ` <span style="color:var(--amber);font-size:10px;font-weight:700">[ROTATE SOON]</span>` : '';
    return `
    <div class="key-row" style="animation-delay:${i * 60}ms">
      <span><span class="dot dot-${status}"></span>${algo} &middot; rotation #${Number(k.rotation_count)}${urgent}</span>
      <span>${Number(k.age_days)}d old &middot; due in ${Number(k.rotation_due_in_days)}d</span>
    </div>`;
  }).join('') || '<div class="empty-state">No keys issued yet.</div>';

  renderHandshakeTrace();

  const logs = await api('/api/security/audit-log', {}, { silent: isPoll });
  document.getElementById('audit-log').innerHTML = logs.map((l, i) => {
    const action   = escapeHtml(String(l.action));
    const resType  = l.resource_type ? ' &middot; ' + escapeHtml(String(l.resource_type)) : '';
    const verified = l.verified
      ? `<span class="verified-badge">✓ ML-DSA</span>`
      : `<span style="color:var(--text-muted);font-size:10px">&mdash;</span>`;
    const ts = l.created_at ? new Date(l.created_at).toLocaleString() : '';
    return `
    <div class="audit-row" style="animation-delay:${i * 30}ms">
      <div class="audit-meta"><span>${escapeHtml(String(l.user_email || ''))}</span><span>${ts}</span></div>
      <div class="audit-action">${action}${resType}</div>
      ${l.signature_preview ? `<div class="audit-sig">${verified} ${escapeHtml(String(l.signature_preview).slice(0,80))}…</div>` : ''}
    </div>`;
  }).join('') || '';
}

async function rotateKeys(algorithm, btn) {
  const confirmed = await confirmAction(
    `Rotate ${algorithm} Keys`,
    `This will immediately invalidate your current ${algorithm} keys and generate a new key pair. All active sessions using the old keys will need to re-authenticate. This cannot be undone.`
  );
  if (!confirmed) return;
  setButtonLoading(btn, true, 'Rotating…');
  try {
    const res = await api('/api/security/rotate-keys', { method: 'POST', body: JSON.stringify({ algorithm, reason: 'manual_rotation' }) });
    await loadSecurity();
    toast('Key rotated', `${algorithm} · rotation #${res.rotation_count} · keygen ${res.keygen_ms} ms`, 'success');
  } catch (err) {
    toast('Rotation failed', err.message, 'error');
  } finally { setButtonLoading(btn, false); }
}
document.getElementById('rotate-dsa').addEventListener('click', (e) => rotateKeys('ML-DSA-65', e.currentTarget));
document.getElementById('rotate-kem').addEventListener('click', (e) => rotateKeys('ML-KEM-768', e.currentTarget));
document.getElementById('compliance-export').addEventListener('click', async () => {
  const btn = document.getElementById('compliance-export');
  setButtonLoading(btn, true, 'Generating…');
  try {
    const report = await api('/api/security/compliance-report');
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'quantumsentinel-compliance-evidence.json'; link.click();
    URL.revokeObjectURL(url); toast('Evidence downloaded', 'Signed audit-verification and key-health report generated.', 'success');
  } catch (err) { toast('Report failed', err.message, 'error'); } finally { setButtonLoading(btn, false); }
});

// ===========================================================================
// Enterprise integrations
// ===========================================================================
function selectedValues(id) {
  const el = document.getElementById(id);
  if (!el) return [];
  // Support both <select multiple> (legacy) and .checkbox-group (new)
  if (el.tagName === 'SELECT') {
    return Array.from(el.selectedOptions).map((o) => o.value);
  }
  return Array.from(el.querySelectorAll('input[type="checkbox"]:checked')).map((cb) => cb.value);
}

async function loadIntegrations() {
  const [keys, hooks] = await Promise.all([
    api('/api/integrations/api-keys', {}, { silent: true }).catch(() => []),
    api('/api/integrations/webhooks', {}, { silent: true }).catch(() => []),
  ]);
  document.getElementById('api-key-list').innerHTML = keys.map((key) =>
    `<div class="order-row"><span>${escapeHtml(key.name)} · ${escapeHtml(key.prefix)}…</span><span>${key.is_revoked ? 'REVOKED' : escapeHtml(key.scopes.join(', '))}</span></div>`
  ).join('') || '<div class="empty-state">No API keys yet.</div>';
  document.getElementById('webhook-list').innerHTML = hooks.map((hook) =>
    `<div class="order-row"><span>${escapeHtml(hook.url)}</span><span>${hook.is_active ? escapeHtml(hook.event_types.join(', ')) : 'DISABLED'}</span></div>`
  ).join('') || '<div class="empty-state">No webhooks yet.</div>';
}

document.getElementById('api-key-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const btn = event.currentTarget.querySelector('button[type=submit]');
  setButtonLoading(btn, true, 'Creating…');
  try {
    const result = await api('/api/integrations/api-keys', { method: 'POST', body: JSON.stringify({
      name: document.getElementById('api-key-name').value, scopes: selectedValues('api-key-scopes'),
    }) });
    const secretEl = document.getElementById('api-key-secret');
    secretEl.textContent = `Copy this API key now; it will not be shown again: ${result.api_key}`;
    // Auto-clear API key from display after 60 seconds for security
    setTimeout(() => { if (secretEl.textContent.includes(result.api_key)) secretEl.textContent = 'API key cleared from display for security.'; }, 60000);
    await loadIntegrations();
  } finally { setButtonLoading(btn, false); }
});

document.getElementById('webhook-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const btn = event.currentTarget.querySelector('button[type=submit]');
  setButtonLoading(btn, true, 'Adding…');
  try {
    const result = await api('/api/integrations/webhooks', { method: 'POST', body: JSON.stringify({
      url: document.getElementById('webhook-url').value, event_types: selectedValues('webhook-events'),
    }) });
    document.getElementById('webhook-secret').textContent = `Copy this signing secret now: ${result.signing_secret}`;
    await loadIntegrations();
  } finally { setButtonLoading(btn, false); }
});

async function loadCommunity() {
  const stats = document.getElementById('community-stats');
  const issuesEl = document.getElementById('community-issues');
  const releasesEl = document.getElementById('community-releases');
  try {
    const repo = 'BugHunterX2101/quantumsentinel-web';
    // Centralised GitHub fetch with rate-limit detection (60 req/hr for anonymous)
    const ghFetch = (url) => fetch(url).then((r) => {
      if (r.status === 403 || r.status === 429) throw new Error('GitHub API rate limit reached — try again in 1 minute');
      if (!r.ok) throw new Error(`GitHub API returned ${r.status}`);
      return r.json();
    });
    const [info, issues, releases] = await Promise.all([
      ghFetch(`https://api.github.com/repos/${repo}`),
      fetch(`https://api.github.com/repos/${repo}/issues?state=open&labels=good%20first%20issue&per_page=8`).then((r) => r.ok ? r.json() : []).catch(() => []),
      fetch(`https://api.github.com/repos/${repo}/releases?per_page=5`).then((r) => r.ok ? r.json() : []).catch(() => []),
    ]);
    stats.innerHTML = '';
    [['Stars', info.stargazers_count], ['Forks', info.forks_count], ['Open Issues', info.open_issues_count]].forEach(([label, value]) => {
      const box = document.createElement('div'); box.className = 'metric-box';
      const val = document.createElement('div'); val.className = 'val'; val.textContent = String(value);
      const lbl = document.createElement('div'); lbl.className = 'lbl'; lbl.textContent = label;
      box.append(val, lbl); stats.append(box);
    });
    function renderItems(container, rows, label) {
      container.replaceChildren();
      if (!rows.length) { container.textContent = `No ${label} available yet.`; return; }
      rows.forEach((row) => { const item = document.createElement('a'); item.className = 'order-row'; item.href = row.html_url; item.target = '_blank'; item.rel = 'noopener noreferrer'; item.textContent = row.title || row.name || row.tag_name; container.append(item); });
    }
    renderItems(issuesEl, issues.filter((issue) => !issue.pull_request), 'good first issues');
    renderItems(releasesEl, releases, 'releases');
  } catch (_) {
    stats.textContent = 'GitHub data is unavailable right now.';
    issuesEl.textContent = ''; releasesEl.textContent = '';
  }
}

// ===========================================================================
// Resume session on page load
// ===========================================================================
(async function init() {
  if (state.token && state.user) {
    // Hide the auth screen immediately — show app shell while handshake runs
    document.getElementById('auth-screen').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');
    document.getElementById('user-email').textContent = state.user.email;
    try {
      await performHandshake({ showOverlay: false });
      await bootstrapApp();
      // Compute actual remaining token lifetime from stored login timestamp
      const loginAt = parseInt(localStorage.getItem('qs_login_at') || '0', 10);
      const expiresIn = parseInt(localStorage.getItem('qs_expires_in') || '900', 10);
      const elapsedSeconds = Math.floor((Date.now() - loginAt) / 1000);
      // Minimum 30s remaining to avoid immediate expiry warning on fresh restores
      const remainingSeconds = Math.max(30, expiresIn - elapsedSeconds);
      scheduleTokenExpiry(remainingSeconds);
    } catch (e) {
      console.error('Session restore failed:', e);
      localStorage.removeItem('qs_token');
      localStorage.removeItem('qs_user');
      location.reload();
    }
  }
})();

// ══════════════════════════════════════════════════════════════════
// RESEARCH ENGINE — Advanced backtest, walk-forward, stat tests
// ══════════════════════════════════════════════════════════════════

let _lastBacktestReturns = null;

// ── Helper: metric card ──
function metricCard(label, value, unit='', cls='') {
  const vStr = typeof value === 'number' ? (Math.abs(value) < 1 && unit === '%' ? (value * 100).toFixed(2) + '%' : value.toFixed(3)) : value;
  return `<div class="metric-card ${cls}"><div class="metric-label">${label}</div><div class="metric-value">${vStr}${unit && typeof value !== 'string' && Math.abs(value) >= 1 ? unit : ''}</div></div>`;
}

function signalBadge(ok, label) {
  return `<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;background:${ok ? 'var(--up-bg,#0d2b1a)' : 'var(--down-bg,#2b0d0d)'};color:${ok ? 'var(--up,#22c55e)' : 'var(--down,#ef4444)'};">${ok ? '✓' : '✗'} ${label}</span>`;
}

// ── Advanced Backtest form ──
document.getElementById('research-backtest-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('rb-error');
  const resultEl = document.getElementById('research-backtest-result');
  errEl.textContent = '';
  resultEl.innerHTML = '<div class="empty-state">Running backtest with execution costs…</div>';

  const assets = document.getElementById('rb-assets').value.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
  const body = {
    assets,
    strategy_type: document.getElementById('rb-strategy').value,
    period: document.getElementById('rb-period').value,
    fast_window: +document.getElementById('rb-fast').value,
    slow_window: +document.getElementById('rb-slow').value,
    execution_preset: document.getElementById('rb-exec').value,
    sizing_method: document.getElementById('rb-sizing').value,
    initial_capital: +document.getElementById('rb-capital').value,
    benchmark: document.getElementById('rb-benchmark').value.trim().toUpperCase() || 'SPY',
    allow_short_selling: document.getElementById('rb-short').checked,
  };

  try {
    const res = await fetch('/api/research/backtest', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
    const d = await res.json();
    _lastBacktestReturns = d.equity_curve_net ? d.equity_curve_net.map((v, i, a) => i > 0 && a[i-1] ? (v - a[i-1]) / a[i-1] : 0).slice(1) : null;
    renderBacktestResult(d, resultEl);
  } catch (err) {
    errEl.textContent = err.message;
    resultEl.innerHTML = '<div class="empty-state">Backtest failed. Check parameters and try again.</div>';
  }
});

function renderBacktestResult(d, el) {
  const retClass = d.total_return >= 0 ? 'up' : 'down';
  const cb = d.cost_breakdown || {};
  let html = `
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px;">
      ${metricCard('Total Return', d.total_return, '%', retClass)}
      ${metricCard('Sharpe (net)', d.sharpe_ratio_net)}
      ${metricCard('Sharpe (gross)', d.sharpe_ratio_gross)}
      ${metricCard('Sortino', d.sortino_ratio)}
      ${metricCard('Calmar', d.calmar_ratio)}
      ${metricCard('Max DD (net)', d.max_drawdown_net, '%')}
      ${metricCard('VaR 95%', d.var_95, '%')}
      ${metricCard('CVaR 95%', d.cvar_95, '%')}
      ${metricCard('Win Rate', d.win_rate, '%')}
      ${metricCard('Total Trades', d.total_trades)}
    </div>`;

  // Cost breakdown
  html += `<h4 style="margin:12px 0 6px;">Transaction Cost Breakdown</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px;">
      ${metricCard('Commission', '$' + (cb.total_commission || 0).toFixed(2))}
      ${metricCard('Slippage', '$' + (cb.total_slippage || 0).toFixed(2))}
      ${metricCard('Spread', '$' + (cb.total_spread || 0).toFixed(2))}
      ${metricCard('Borrow', '$' + (cb.total_borrow || 0).toFixed(2))}
      ${metricCard('Total Costs', '$' + (cb.total_costs || 0).toFixed(2))}
      ${metricCard('Costs % Capital', (cb.costs_pct_of_capital || 0).toFixed(2) + '%')}
    </div>`;

  // Benchmark comparison
  if (d.benchmark) {
    html += `<h4 style="margin:12px 0 6px;">vs Benchmark (${d.benchmark.ticker})</h4>
      <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px;">
        ${metricCard('Bench Return', d.benchmark.total_return, '%')}
        ${metricCard('Bench Sharpe', d.benchmark.sharpe)}
        ${metricCard('Alpha', d.alpha || 0)}
        ${metricCard('Beta', d.beta || 0)}
        ${metricCard('Info Ratio', d.information_ratio || 0)}
        ${metricCard('Track Error', d.tracking_error || 0)}
      </div>`;
  }

  // Mini equity chart (ASCII sparkline)
  if (d.equity_curve_net && d.equity_curve_net.length > 1) {
    const curve = d.equity_curve_net;
    const mn = Math.min(...curve), mx = Math.max(...curve);
    const bars = curve.map(v => {
      const h = mx > mn ? Math.round(((v - mn) / (mx - mn)) * 40) + 2 : 20;
      return `<div style="flex:1;min-width:2px;height:${h}px;background:var(--accent);border-radius:1px 1px 0 0;"></div>`;
    }).join('');
    html += `<h4 style="margin:12px 0 6px;">Equity Curve (Net of Costs)</h4>
      <div style="display:flex;align-items:flex-end;height:50px;gap:1px;padding:4px 0;">${bars}</div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-3);"><span>$${curve[0].toLocaleString()}</span><span>$${curve[curve.length-1].toLocaleString()}</span></div>`;
  }

  html += `<div style="margin-top:12px;font-size:11px;color:var(--text-3);">Executed in ${d.execution_time_ms}ms</div>`;
  el.innerHTML = html;
}

// ── Walk-Forward form ──
document.getElementById('research-wf-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('wf-error');
  const resultEl = document.getElementById('research-wf-result');
  errEl.textContent = '';
  resultEl.innerHTML = '<div class="empty-state">Running walk-forward validation (this may take 30-60s)…</div>';

  const body = {
    assets: [document.getElementById('wf-asset').value.trim().toUpperCase()],
    window_type: document.getElementById('wf-type').value,
    total_years: +document.getElementById('wf-total').value,
    train_years: +document.getElementById('wf-train').value,
    test_years: +document.getElementById('wf-test').value,
    optimize_parameters: document.getElementById('wf-optimize').checked,
  };

  try {
    const res = await fetch('/api/research/walk-forward', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
    const d = await res.json();
    renderWalkForwardResult(d, resultEl);
  } catch (err) {
    errEl.textContent = err.message;
    resultEl.innerHTML = '<div class="empty-state">Walk-forward failed. Check parameters.</div>';
  }
});

function renderWalkForwardResult(d, el) {
  const agg = d.aggregated_oos || {};
  const of = d.overfitting_analysis || {};
  const ps = d.parameter_stability || {};

  let html = `
    <h4 style="margin:0 0 8px;">Aggregated Out-of-Sample (${d.n_folds} folds, ${d.window_type})</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:16px;">
      ${metricCard('OOS Sharpe', agg.sharpe)}
      ${metricCard('OOS Sortino', agg.sortino)}
      ${metricCard('OOS Return', agg.total_return, '%')}
      ${metricCard('OOS Max DD', agg.max_drawdown, '%')}
      ${metricCard('OOS Calmar', agg.calmar)}
      ${metricCard('OOS CVaR 95', agg.cvar_95, '%')}
    </div>`;

  // Overfitting analysis
  html += `<h4 style="margin:12px 0 6px;">Overfitting Analysis</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:10px;">
      ${metricCard('Avg Train Sharpe', of.avg_train_sharpe)}
      ${metricCard('Avg OOS Sharpe', of.avg_oos_sharpe)}
      ${metricCard('Sharpe Decay', of.sharpe_decay)}
      ${metricCard('Overfit Score', of.overfitting_score)}
    </div>
    <div style="margin-bottom:12px;">
      ${signalBadge(!of.likely_overfit, of.likely_overfit ? 'Likely Overfit — OOS performance decays significantly' : 'No strong overfitting signal — OOS performance holds')}
    </div>`;

  // Parameter stability
  if (ps.fast_windows) {
    html += `<h4 style="margin:12px 0 6px;">Parameter Stability</h4>
      <div style="font-size:13px;margin-bottom:6px;">
        <span style="color:var(--text-2);">Fast windows:</span> ${ps.fast_windows.join(', ')}
        <span style="margin-left:12px;color:var(--text-2);">Slow windows:</span> ${ps.slow_windows.join(', ')}
      </div>
      ${signalBadge(ps.parameters_stable, ps.parameters_stable ? 'Parameters stable across folds' : 'Parameters vary — possible instability')}`;
  }

  // Per-fold table
  if (d.folds && d.folds.length > 0) {
    html += `<h4 style="margin:16px 0 6px;">Per-Fold Results</h4>
      <div style="overflow-x:auto;"><table style="width:100%;font-size:12px;border-collapse:collapse;">
        <thead><tr style="border-bottom:1px solid var(--border);">
          <th style="padding:4px 8px;text-align:left;">Fold</th>
          <th style="padding:4px 8px;">Train Sharpe</th>
          <th style="padding:4px 8px;">OOS Sharpe</th>
          <th style="padding:4px 8px;">OOS Return</th>
          <th style="padding:4px 8px;">OOS Max DD</th>
          <th style="padding:4px 8px;">Best Fast</th>
          <th style="padding:4px 8px;">Best Slow</th>
        </tr></thead><tbody>`;
    for (const f of d.folds) {
      html += `<tr style="border-bottom:1px solid var(--border);">
        <td style="padding:4px 8px;">${f.fold + 1}</td>
        <td style="padding:4px 8px;text-align:center;">${f.train_sharpe.toFixed(3)}</td>
        <td style="padding:4px 8px;text-align:center;color:${f.oos_sharpe > 0 ? 'var(--up)' : 'var(--down)'}">${f.oos_sharpe.toFixed(3)}</td>
        <td style="padding:4px 8px;text-align:center;">${(f.oos_return * 100).toFixed(2)}%</td>
        <td style="padding:4px 8px;text-align:center;">${(f.oos_max_dd * 100).toFixed(2)}%</td>
        <td style="padding:4px 8px;text-align:center;">${f.best_fast_window}</td>
        <td style="padding:4px 8px;text-align:center;">${f.best_slow_window}</td>
      </tr>`;
    }
    html += '</tbody></table></div>';
  }

  html += `<div style="margin-top:12px;font-size:11px;color:var(--text-3);">Completed in ${d.execution_time_ms}ms</div>`;
  el.innerHTML = html;
}

// ── Statistical Tests form ──
document.getElementById('research-stat-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('st-error');
  const resultEl = document.getElementById('research-stat-result');
  errEl.textContent = '';

  if (!_lastBacktestReturns || _lastBacktestReturns.length < 5) {
    errEl.textContent = 'Run an advanced backtest first to generate return data.';
    return;
  }

  resultEl.innerHTML = '<div class="empty-state">Running statistical tests (bootstrap + permutation)…</div>';

  const body = {
    returns: _lastBacktestReturns,
    n_strategies_tested: +document.getElementById('st-trials').value,
  };

  try {
    const res = await fetch('/api/research/stat-test', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
    const d = await res.json();
    renderStatResult(d, resultEl);
  } catch (err) {
    errEl.textContent = err.message;
    resultEl.innerHTML = '<div class="empty-state">Statistical tests failed.</div>';
  }
});

function renderStatResult(d, el) {
  const summ = d.summary || {};
  const tt = d.ttest_nw || {};
  const bs = d.bootstrap_sharpe || {};
  const pm = d.permutation_test || {};
  const dsr = d.deflated_sharpe || {};
  const lb = d.ljung_box || {};

  let html = `
    <h4 style="margin:0 0 10px;">Overall Verdict</h4>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
      ${signalBadge(summ.overall_credible, summ.overall_credible ? 'Strategy is statistically credible' : 'Strategy does NOT pass all significance tests')}
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;">
      ${signalBadge(summ.mean_return_significant, 'Mean Return ≠ 0')}
      ${signalBadge(summ.sharpe_ci_excludes_zero, 'Sharpe CI > 0')}
      ${signalBadge(summ.permutation_significant, 'Permutation Test')}
      ${signalBadge(summ.survives_deflation, 'Deflated Sharpe')}
      ${signalBadge(!summ.has_autocorrelation, 'No Autocorrelation')}
    </div>`;

  // t-test
  html += `<h4 style="margin:12px 0 6px;">t-Test (Newey-West HAC)</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:12px;">
      ${metricCard('t-Statistic', tt.t_stat)}
      ${metricCard('p-Value', tt.p_value)}
      ${metricCard('Ann. Mean', tt.annualized_mean)}
      ${metricCard('N', tt.n)}
    </div>`;

  // Bootstrap
  html += `<h4 style="margin:12px 0 6px;">Bootstrap Sharpe (${bs.n_bootstrap?.toLocaleString()} samples)</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:12px;">
      ${metricCard('Sharpe', bs.sharpe)}
      ${metricCard('95% CI Lower', bs.ci_lower)}
      ${metricCard('95% CI Upper', bs.ci_upper)}
      ${metricCard('P(Sharpe > 0)', bs.probability_positive, '%')}
    </div>`;

  // Permutation
  html += `<h4 style="margin:12px 0 6px;">Permutation Test</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:12px;">
      ${metricCard('Observed SR', pm.observed_sharpe)}
      ${metricCard('p-Value', pm.p_value)}
      ${metricCard('Exceeded By', pm.count_exceeding + '/' + pm.n_permutations)}
    </div>`;

  // Deflated Sharpe
  html += `<h4 style="margin:12px 0 6px;">Deflated Sharpe Ratio (Bailey & López de Prado)</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:8px;">
      ${metricCard('Observed SR', dsr.observed_sharpe)}
      ${metricCard('E[max(SR)]', dsr.expected_max_sharpe)}
      ${metricCard('DSR p-Value', dsr.dsr_p_value)}
      ${metricCard('# Trials', dsr.n_trials)}
      ${metricCard('Haircut %', dsr.haircut_pct + '%')}
    </div>
    <p style="font-size:12px;color:var(--text-2);margin:4px 0 12px;">${dsr.interpretation || ''}</p>`;

  // Autocorrelation
  html += `<h4 style="margin:12px 0 6px;">Ljung-Box Autocorrelation</h4>
    <div class="metrics-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:8px;">
      ${metricCard('Q-Statistic', lb.test_statistic)}
      ${metricCard('p-Value', lb.p_value)}
    </div>
    <p style="font-size:12px;color:var(--text-2);">${lb.interpretation || ''}</p>`;

  el.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════
// PHASE 2 — Alpha Research, Factor Model, Correlation, Port Opt
// ══════════════════════════════════════════════════════════════════

function parseAssets(inputEl) {
  return inputEl.value.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
}

// ── Alpha Research ──────────────────────────────────────────────
document.getElementById('alpha-research-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('ar-error');
  const resultEl = document.getElementById('alpha-research-result');
  errEl.textContent = '';
  resultEl.innerHTML = '<div class="empty-state">Running alpha research (computing IC, decay, quintile analysis)…</div>';

  const body = {
    assets: parseAssets(document.getElementById('ar-assets')),
    signal_type: document.getElementById('ar-signal').value,
    period: document.getElementById('ar-period').value,
    max_horizon: +document.getElementById('ar-horizon').value,
  };

  try {
    const res = await fetch('/api/research/alpha', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
    renderAlphaResult(await res.json(), resultEl);
  } catch (err) {
    errEl.textContent = err.message;
    resultEl.innerHTML = '<div class="empty-state">Alpha research failed.</div>';
  }
});

function renderAlphaResult(d, el) {
  const ic = d.ic_analysis || {};
  const decay = d.ic_decay || {};
  const hr = d.hit_rate || {};
  const quint = d.quintile_analysis || {};
  const to = d.factor_turnover || {};
  const qs = d.alpha_quality_score || {};

  // Quality score pill
  const scoreColor = qs.score >= 75 ? 'var(--up)' : qs.score >= 50 ? '#f59e0b' : qs.score >= 25 ? '#f97316' : 'var(--down)';
  let html = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <div style="width:52px;height:52px;border-radius:50%;background:conic-gradient(${scoreColor} ${qs.score * 3.6}deg,var(--surface-2) 0);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:${scoreColor};">${qs.score || 0}</div>
      <div>
        <div style="font-size:16px;font-weight:700;color:${scoreColor};">${qs.rating || 'N/A'} Alpha Quality</div>
        <div style="font-size:12px;color:var(--text-3);">${d.n_assets} assets · ${d.n_periods} periods · ${d.signal_type} signal</div>
      </div>
    </div>`;

  // IC metrics grid
  html += `<h4 style="margin:0 0 8px;">Information Coefficient</h4>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-bottom:14px;">
      ${metricCard('Mean IC', ic.mean_ic)}
      ${metricCard('IC Std', ic.ic_std)}
      ${metricCard('ICIR', ic.icir)}
      ${metricCard('t-Stat', ic.t_stat)}
      ${metricCard('p-Value', ic.p_value)}
      ${metricCard('+IC %', ic.positive_ic_pct, '%')}
    </div>
    <div style="margin-bottom:10px;">${signalBadge(ic.significant_5pct, ic.significant_5pct ? 'IC statistically significant (p<5%)' : 'IC not significant — signal may lack predictive power')}</div>
    <p style="font-size:12px;color:var(--text-2);margin-bottom:14px;">${ic.interpretation || ''}</p>`;

  // IC decay sparkline
  if (decay.ic_by_horizon && decay.ic_by_horizon.length > 0) {
    const vals = decay.ic_by_horizon.map(x => x.mean_ic);
    const mn = Math.min(...vals), mx = Math.max(Math.abs(mn), ...vals.map(Math.abs));
    const bars = vals.map((v, i) => {
      const h = mx > 0 ? Math.abs(v) / mx * 36 + 2 : 4;
      const c = v >= 0 ? 'var(--accent)' : 'var(--down)';
      return `<div style="display:flex;flex-direction:column;align-items:center;flex:1;"><div style="width:100%;height:${h}px;background:${c};border-radius:2px;"></div><div style="font-size:9px;color:var(--text-3);">${decay.horizons[i]}d</div></div>`;
    }).join('');
    html += `<h4 style="margin:0 0 6px;">IC Decay (alpha half-life: ${decay.half_life_days || '> ' + decay.horizons.at(-1)} days)</h4>
      <div style="display:flex;align-items:flex-end;gap:2px;height:48px;margin-bottom:14px;">${bars}</div>`;
  }

  // Hit rate + quintile spread
  html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
    <div>
      <h4 style="margin:0 0 6px;">Hit Rate</h4>
      ${metricCard('Hit Rate', hr.hit_rate, '%')}
      <div style="margin-top:6px;">${signalBadge(hr.significant_5pct, hr.significant_5pct ? 'Directionally significant' : 'Not directionally significant')}</div>
    </div>
    <div>
      <h4 style="margin:0 0 6px;">Quintile Spread</h4>
      ${metricCard('Ann. Spread', quint.annualised_spread, '%')}
      <div style="margin-top:6px;">${signalBadge(quint.monotonic, quint.monotonic ? 'Monotonic Q1→Q5' : 'Non-monotonic quintiles')}</div>
    </div>
  </div>`;

  // Quintile bar chart
  if (quint.quantile_returns && quint.quantile_returns.length > 0) {
    const qvals = quint.quantile_returns;
    const qmax = Math.max(...qvals.map(Math.abs));
    const qbars = qvals.map((v, i) => {
      const h = qmax > 0 ? Math.abs(v) / qmax * 32 + 4 : 4;
      const c = v >= 0 ? 'var(--up)' : 'var(--down)';
      return `<div style="display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;">
        <div style="width:80%;height:${h}px;background:${c};border-radius:2px;"></div>
        <div style="font-size:10px;color:var(--text-3);">Q${i+1}</div>
      </div>`;
    }).join('');
    html += `<h4 style="margin:0 0 6px;">Quintile Returns</h4>
      <div style="display:flex;align-items:flex-end;gap:4px;height:44px;margin-bottom:12px;">${qbars}</div>`;
  }

  // Factor turnover
  html += `<h4 style="margin:0 0 6px;">Factor Turnover</h4>
    <div style="font-size:13px;">Avg rank correlation: <b>${to.avg_rank_correlation}</b> · Turnover: <b>${to.avg_turnover}</b></div>
    <div style="font-size:12px;color:var(--text-2);margin-top:4px;">${to.turnover_interpretation || ''}</div>`;

  el.innerHTML = html;
}

// ── Factor Model (Fama-MacBeth) ───────────────────────────────────
document.getElementById('factor-model-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('fm-error');
  const resultEl = document.getElementById('factor-model-result');
  errEl.textContent = '';
  resultEl.innerHTML = '<div class="empty-state">Running Fama-MacBeth regression (this may take 30-60s)…</div>';

  const factors = [...document.querySelectorAll('#fm-factors input:checked')].map(c => c.value);
  if (factors.length === 0) { errEl.textContent = 'Select at least one factor.'; return; }

  const body = {
    assets: parseAssets(document.getElementById('fm-assets')),
    period: document.getElementById('fm-period').value,
    factors,
    newey_west_lags: +document.getElementById('fm-nw').value,
  };

  try {
    const res = await fetch('/api/research/factor-model', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
    renderFactorModelResult(await res.json(), resultEl);
  } catch (err) {
    errEl.textContent = err.message;
    resultEl.innerHTML = '<div class="empty-state">Factor model failed.</div>';
  }
});

function renderFactorModelResult(d, el) {
  const fm = d.fama_macbeth || {};
  const risk = d.risk_decomposition || {};
  if (fm.error) { el.innerHTML = `<div class="empty-state">${fm.error}</div>`; return; }

  const premia = fm.factor_premia || {};
  const ranked = fm.factors_ranked_by_significance || [];

  let html = `<h4 style="margin:0 0 8px;">Factor Risk Premia (${d.asset_names?.length || 0} assets, ${fm.n_cross_sections} cross-sections)</h4>
    <p style="font-size:12px;color:var(--text-3);margin-bottom:10px;">Cross-sectional R² = ${(fm.mean_cross_sectional_r2 * 100).toFixed(1)}%</p>
    <div style="overflow-x:auto;"><table style="width:100%;font-size:12px;border-collapse:collapse;">
      <thead><tr style="border-bottom:1px solid var(--border);">
        <th style="padding:5px 8px;text-align:left;">Factor</th>
        <th style="padding:5px 8px;">λ (daily)</th>
        <th style="padding:5px 8px;">λ (annual)</th>
        <th style="padding:5px 8px;">NW t-stat</th>
        <th style="padding:5px 8px;">p-value</th>
        <th style="padding:5px 8px;">Significant</th>
      </tr></thead><tbody>`;

  for (const { factor } of ranked) {
    const p = premia[factor];
    if (!p) continue;
    const sigColor = p.significant_1pct ? 'var(--up)' : p.significant_5pct ? '#f59e0b' : 'var(--text-3)';
    const sigLabel = p.significant_1pct ? '✓ 1%' : p.significant_5pct ? '✓ 5%' : '✗';
    html += `<tr style="border-bottom:1px solid var(--border);">
      <td style="padding:5px 8px;font-weight:600;">${factor}</td>
      <td style="padding:5px 8px;text-align:center;color:${p.lambda >= 0 ? 'var(--up)' : 'var(--down)'};">${p.lambda?.toFixed(5)}</td>
      <td style="padding:5px 8px;text-align:center;color:${p.lambda_annualised >= 0 ? 'var(--up)' : 'var(--down)'};">${(p.lambda_annualised * 100)?.toFixed(2)}%</td>
      <td style="padding:5px 8px;text-align:center;">${p.t_stat?.toFixed(3)}</td>
      <td style="padding:5px 8px;text-align:center;">${p.p_value?.toFixed(4)}</td>
      <td style="padding:5px 8px;text-align:center;font-weight:700;color:${sigColor};">${sigLabel}</td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  // Barra risk decomposition
  if (!risk.error) {
    html += `<h4 style="margin:16px 0 8px;">Barra Risk Decomposition</h4>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;">
        ${metricCard('Factor Risk', (risk.pct_factor_explained * 100)?.toFixed(1) + '%')}
        ${metricCard('Specific Risk', (risk.pct_specific * 100)?.toFixed(1) + '%')}
        ${metricCard('Avg Factor Var', risk.avg_factor_variance?.toFixed(4))}
        ${metricCard('Avg Specific Var', risk.avg_specific_variance?.toFixed(4))}
        ${metricCard('Valid Assets', risk.n_valid_assets)}
      </div>`;
  }

  el.innerHTML = html;
}

// ── Correlation Engine ───────────────────────────────────────────
document.getElementById('correlation-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('ce-error');
  const resultEl = document.getElementById('correlation-result');
  errEl.textContent = '';
  resultEl.innerHTML = '<div class="empty-state">Computing correlations across 6 estimators…</div>';

  const body = {
    assets: parseAssets(document.getElementById('ce-assets')),
    period: document.getElementById('ce-period').value,
    ewma_halflife: +document.getElementById('ce-halflife').value,
  };

  try {
    const res = await fetch('/api/research/correlation', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
    renderCorrelationResult(await res.json(), resultEl);
  } catch (err) {
    errEl.textContent = err.message;
    resultEl.innerHTML = '<div class="empty-state">Correlation analysis failed.</div>';
  }
});

function renderCorrelationResult(d, el) {
  const diag = d.diagnostics || {};
  const rec = d.recommended_estimator || {};
  const lw = d.shrinkage_intensities || {};
  const pca = d.pca_info || {};
  const names = d.ticker_names || [];

  let html = `<h4 style="margin:0 0 8px;">Recommendation</h4>
    <div style="padding:10px;background:var(--surface-2);border-radius:8px;margin-bottom:14px;">
      <span style="font-weight:700;color:var(--accent);">${rec.estimator?.toUpperCase()}</span>
      <span style="font-size:12px;color:var(--text-2);margin-left:8px;">${rec.reason || ''}</span>
    </div>
    <p style="font-size:12px;color:var(--text-3);margin-bottom:10px;">T=${d.n_periods} · N=${d.n_assets} · T/N=${d.ratio_T_over_N}</p>`;

  // Diagnostics table
  html += `<h4 style="margin:0 0 6px;">Estimator Diagnostics</h4>
    <div style="overflow-x:auto;"><table style="width:100%;font-size:12px;border-collapse:collapse;">
      <thead><tr style="border-bottom:1px solid var(--border);">
        <th style="padding:4px 8px;text-align:left;">Method</th>
        <th style="padding:4px 8px;">Positive Definite</th>
        <th style="padding:4px 8px;">Condition #</th>
        <th style="padding:4px 8px;">Avg |Corr|</th>
        <th style="padding:4px 8px;">Eff. Bets</th>
      </tr></thead><tbody>`;

  const methodLabels = {
    pearson: 'Pearson', spearman: 'Spearman', ewma: `EWMA ${d.pca_info ? '' : ''}`,
    ledoit_wolf: 'Ledoit-Wolf', oas: 'OAS', pca: `PCA (${pca.n_components}f)`,
  };

  for (const [m, info] of Object.entries(diag)) {
    const pdColor = info.is_positive_definite ? 'var(--up)' : 'var(--down)';
    const isRec = m === rec.estimator;
    html += `<tr style="border-bottom:1px solid var(--border);${isRec ? 'background:rgba(99,102,241,0.07);' : ''}">
      <td style="padding:4px 8px;font-weight:${isRec ? '700' : 'normal'};">${methodLabels[m] || m}${isRec ? ' ★' : ''}</td>
      <td style="padding:4px 8px;text-align:center;color:${pdColor};">${info.is_positive_definite ? '✓' : '✗'}</td>
      <td style="padding:4px 8px;text-align:center;">${info.condition_number?.toLocaleString()}</td>
      <td style="padding:4px 8px;text-align:center;">${info.avg_abs_correlation?.toFixed(3)}</td>
      <td style="padding:4px 8px;text-align:center;">${info.effective_uncorrelated_bets?.toFixed(1)}</td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  // Shrinkage info
  html += `<div style="margin-top:12px;font-size:12px;color:var(--text-2);">
    Ledoit-Wolf intensity: <b>${lw.ledoit_wolf}</b> &nbsp;|&nbsp; OAS intensity: <b>${lw.oas}</b>
    &nbsp;|&nbsp; PCA components: <b>${pca.n_components}</b> (${pca.cumulative_explained?.at(-1) * 100 | 0}% var explained)
  </div>`;

  // Mini correlation heatmap for recommended estimator (text-based)
  const recCorr = d.correlations?.[rec.estimator];
  if (recCorr && names.length <= 12) {
    const N = names.length;
    html += `<h4 style="margin:14px 0 6px;">${methodLabels[rec.estimator] || rec.estimator} Correlation Matrix</h4>
      <div style="overflow-x:auto;"><table style="font-size:10px;border-collapse:collapse;">
        <thead><tr><th></th>${names.map(n => `<th style="padding:2px 4px;transform:rotate(-30deg);min-width:28px;">${n}</th>`).join('')}</tr></thead><tbody>`;
    for (let i = 0; i < N; i++) {
      html += `<tr><td style="padding:2px 4px;font-weight:600;">${names[i]}</td>`;
      for (let j = 0; j < N; j++) {
        const v = recCorr[i]?.[j] ?? 0;
        const abs = Math.abs(v);
        const alpha = i === j ? 0.15 : abs * 0.7;
        const bg = v > 0 ? `rgba(99,102,241,${alpha})` : `rgba(239,68,68,${alpha})`;
        html += `<td style="padding:2px 4px;text-align:center;background:${bg};border-radius:2px;">${i === j ? '—' : v.toFixed(2)}</td>`;
      }
      html += '</tr>';
    }
    html += '</tbody></table></div>';
  }

  el.innerHTML = html;
}

// ── Portfolio Optimisation ───────────────────────────────────────
document.getElementById('portopt-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('po-error');
  const resultEl = document.getElementById('portopt-result');
  errEl.textContent = '';
  resultEl.innerHTML = '<div class="empty-state">Running portfolio optimisation across all methods…</div>';

  const body = {
    assets: parseAssets(document.getElementById('po-assets')),
    period: document.getElementById('po-period').value,
    covariance_method: document.getElementById('po-cov').value,
    long_only: document.getElementById('po-longonly').checked,
    max_weight: +document.getElementById('po-maxw').value / 100,
    min_weight: 0,
    risk_free_rate: +document.getElementById('po-rf').value / 100,
    include_sba: document.getElementById('po-sba').checked,
  };

  try {
    const res = await fetch('/api/research/optimize', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || res.statusText); }
    renderPortOptResult(await res.json(), resultEl);
  } catch (err) {
    errEl.textContent = err.message;
    resultEl.innerHTML = '<div class="empty-state">Optimisation failed.</div>';
  }
});

function renderPortOptResult(d, el) {
  const portfolios = d.portfolios || {};
  const ranked = d.portfolios_ranked_by_sharpe || [];
  const frontier = d.efficient_frontier || [];

  const methodNames = {
    equal_weight: 'Equal Weight', min_variance: 'Min Variance',
    max_sharpe: 'Max Sharpe', risk_parity: 'Risk Parity',
    max_diversification: 'Max Diversification', sba_signal: 'SBA Signal',
  };

  let html = `<h4 style="margin:0 0 8px;">Method Comparison (${d.n_assets} assets, ${d.covariance_method})</h4>
    <div style="overflow-x:auto;"><table style="width:100%;font-size:12px;border-collapse:collapse;">
      <thead><tr style="border-bottom:1px solid var(--border);">
        <th style="padding:5px 8px;text-align:left;">Method</th>
        <th style="padding:5px 8px;">Ann. Return</th>
        <th style="padding:5px 8px;">Ann. Vol</th>
        <th style="padding:5px 8px;">Sharpe</th>
        <th style="padding:5px 8px;">Div. Ratio</th>
        <th style="padding:5px 8px;">Eff. N</th>
        <th style="padding:5px 8px;">Max Wt</th>
      </tr></thead><tbody>`;

  for (const { method } of ranked) {
    const p = portfolios[method];
    if (!p) continue;
    const isTop = ranked[0]?.method === method;
    html += `<tr style="border-bottom:1px solid var(--border);${isTop ? 'background:rgba(99,102,241,0.07);' : ''}">
      <td style="padding:5px 8px;font-weight:${isTop ? '700' : 'normal'};">${methodNames[method] || method}${isTop ? ' ★' : ''}</td>
      <td style="padding:5px 8px;text-align:center;color:${p.annual_return >= 0 ? 'var(--up)' : 'var(--down)'};">${(p.annual_return * 100).toFixed(2)}%</td>
      <td style="padding:5px 8px;text-align:center;">${(p.annual_volatility * 100).toFixed(2)}%</td>
      <td style="padding:5px 8px;text-align:center;color:${p.sharpe_ratio > 0 ? 'var(--up)' : 'var(--down)'};">${p.sharpe_ratio?.toFixed(3)}</td>
      <td style="padding:5px 8px;text-align:center;">${p.diversification_ratio?.toFixed(2)}</td>
      <td style="padding:5px 8px;text-align:center;">${p.effective_n?.toFixed(1)}</td>
      <td style="padding:5px 8px;text-align:center;">${(p.max_weight * 100).toFixed(1)}%</td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  // Best method weights as bar chart
  const bestMethod = ranked[0]?.method;
  const bestPortfolio = portfolios[bestMethod];
  if (bestPortfolio?.weights) {
    const entries = Object.entries(bestPortfolio.weights).sort((a, b) => b[1] - a[1]).slice(0, 12);
    const maxW = Math.max(...entries.map(x => x[1]));
    html += `<h4 style="margin:16px 0 8px;">Top Holdings — ${methodNames[bestMethod] || bestMethod}</h4>
      <div style="display:flex;flex-direction:column;gap:4px;margin-bottom:14px;">`;
    for (const [ticker, wt] of entries) {
      const barW = maxW > 0 ? (wt / maxW * 100).toFixed(1) : 0;
      html += `<div style="display:flex;align-items:center;gap:8px;font-size:12px;">
        <span style="min-width:48px;color:var(--text-2);">${ticker}</span>
        <div style="flex:1;height:14px;background:var(--surface-2);border-radius:3px;overflow:hidden;">
          <div style="width:${barW}%;height:100%;background:var(--accent);border-radius:3px;"></div>
        </div>
        <span style="min-width:38px;text-align:right;font-weight:600;">${(wt * 100).toFixed(1)}%</span>
      </div>`;
    }
    html += '</div>';
  }

  // Efficient frontier mini-chart
  if (frontier.length > 0) {
    const vols = frontier.map(p => p.volatility);
    const rets = frontier.map(p => p.return);
    const minV = Math.min(...vols), maxV = Math.max(...vols);
    const minR = Math.min(...rets), maxR = Math.max(...rets);

    html += `<h4 style="margin:0 0 6px;">Efficient Frontier</h4>
      <div style="position:relative;height:80px;background:var(--surface-2);border-radius:6px;padding:4px;margin-bottom:8px;overflow:hidden;">
        <svg width="100%" height="100%" viewBox="0 0 300 80" preserveAspectRatio="none">
          <polyline points="${frontier.map(p => {
            const x = maxV > minV ? ((p.volatility - minV) / (maxV - minV)) * 290 + 5 : 150;
            const y = maxR > minR ? 75 - ((p.return - minR) / (maxR - minR)) * 70 : 40;
            return `${x},${y}`;
          }).join(' ')}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>
        </svg>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-3);">
        <span>Vol: ${(minV * 100).toFixed(1)}%</span><span>← Risk/Return →</span><span>Vol: ${(maxV * 100).toFixed(1)}%</span>
      </div>`;
  }

  el.innerHTML = html;
}
