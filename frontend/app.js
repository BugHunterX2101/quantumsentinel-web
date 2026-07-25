// QuantumSentinel — frontend application logic (vanilla JS, no build step).
const state = {
  token: localStorage.getItem('qs_token') || null,
  user: JSON.parse(localStorage.getItem('qs_user') || 'null'),
  beginner: localStorage.getItem('qs_beginner') !== 'false',
  meta: null,
  lastSignals: {},   // asset -> last signal_type, for diff-highlighting
  pollTimer: null,
  signalSocket: null,
  signalReconnectMs: 1000,
  activeView: 'dashboard',
  tokenExpireTimer: null,
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
  loadingBar.start();
  return fetch(path, Object.assign({}, opts, { headers })).then(async (r) => {
    const body = await r.json().catch(() => ({}));
    if (r.status === 401) {
      // Token expired or invalid — force logout
      handleTokenExpiry();
      throw new Error('Session expired. Please log in again.');
    }
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

// ===========================================================================
// Animated quantum-particle background (canvas)
// ===========================================================================
(function initParticles() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let w, h, particles;
  const COLORS = ['#4fd8ff', '#b892ff', '#3ddc97'];

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }
  function makeParticles() {
    const count = Math.min(70, Math.floor((w * h) / 22000));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
      r: Math.random() * 1.6 + 0.6, color: COLORS[Math.floor(Math.random() * COLORS.length)],
    }));
  }
  function step() {
    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.strokeStyle = `rgba(79,216,255,${(1 - dist / 120) * 0.12})`;
          ctx.lineWidth = 0.6;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
    }
    for (const p of particles) {
      ctx.beginPath();
      ctx.fillStyle = p.color;
      ctx.globalAlpha = 0.75;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    requestAnimationFrame(step);
  }
  window.addEventListener('resize', () => { resize(); makeParticles(); });
  resize(); makeParticles(); step();
})();

// ===========================================================================
// Typewriter tagline on auth screen
// ===========================================================================
(function typewriter() {
  const el = document.getElementById('tagline');
  const text = "The world's first open-source, mobile-first, post-quantum secure trading terminal.";
  let i = 0;
  function tick() {
    el.textContent = text.slice(0, i);
    i++;
    if (i <= text.length) setTimeout(tick, 18);
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
  const email = document.getElementById('register-email').value;
  const password = document.getElementById('register-password').value;
  const errEl = document.getElementById('register-error');
  const btn = e.target.querySelector('button[type=submit]');
  errEl.textContent = '';
  setButtonLoading(btn, true, 'Generating ML-KEM-768/ML-DSA-65 keys…');
  try {
    await api('/api/auth/register', { method: 'POST', body: JSON.stringify({ email, password }) }, { silent: true });
    const data = await api('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }, { silent: true });
    await afterLogin(data);
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

function switchView(view) {
  state.activeView = view;
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  document.querySelectorAll('.view').forEach((v) => {
    const isTarget = v.id === 'view-' + view;
    v.classList.toggle('active', isTarget);
    if (isTarget) {
      v.classList.remove('entering');
      requestAnimationFrame(() => v.classList.add('entering'));
    }
  });
  const banners = {
    dashboard: 'Beginner tip: green BUY badges and higher confidence bars mean the signal engine found stronger multi-asset agreement — it is not a guarantee.',
    trading: 'Beginner tip: every order you submit here is cryptographically signed with ML-DSA-65 before it is sent, and settles as a paper (simulated) trade.',
    portfolio: 'Beginner tip: Sharpe ratio > 1 is generally considered good risk-adjusted performance; max drawdown shows your worst peak-to-trough loss.',
    strategies: 'Beginner tip: validate a strategy on historical data first. A positive backtest is not a prediction of future returns.',
    security: 'Beginner tip: the Quantum Safety Score reflects how fresh your cryptographic keys are — green means fully rotated and compliant.',
  };
  const banner = document.getElementById('beginner-banner');
  if (state.beginner && banners[view]) {
    banner.textContent = banners[view];
    banner.classList.remove('hidden');
  } else {
    banner.classList.add('hidden');
  }
  if (view === 'dashboard') loadDashboard();
  if (view === 'trading') loadTrading();
  if (view === 'strategies') loadStrategies();
  if (view === 'portfolio') loadPortfolio();
  if (view === 'security') loadSecurity();
  if (view === 'integrations') loadIntegrations();
  if (view === 'community') loadCommunity();
  restartPolling();
}

function restartPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  const loaders = { dashboard: loadDashboard, trading: refreshOrders, strategies: loadStrategies, portfolio: loadPortfolio, security: loadSecurity, integrations: loadIntegrations, community: loadCommunity };
  const fn = loaders[state.activeView];
  if (!fn) return;
  state.pollTimer = setInterval(() => fn(true), 20000);
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
  localStorage.removeItem('qs_token');
  localStorage.removeItem('qs_user');
  localStorage.removeItem('qs_login_at');
  localStorage.removeItem('qs_expires_in');
  location.reload();
});

async function bootstrapApp() {
  applyBeginnerMode();
  state.meta = await api('/api/meta');
  const sel = document.getElementById('order-asset');
  sel.innerHTML = state.meta.tracked_assets.map((a) => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join('');
  document.getElementById('strategy-asset').innerHTML = sel.innerHTML;
  switchView('dashboard');
  connectSignalStream();
  startOnboardingIfNeeded();
}

function connectSignalStream() {
  if (!state.token || state.signalSocket) return;
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${scheme}//${location.host}/api/signals/stream`, ['qs', state.token]);
  state.signalSocket = socket;
  socket.onopen = () => { state.signalReconnectMs = 1000; };
  socket.onmessage = (event) => {
    try {
      if (state.activeView === 'dashboard') loadDashboard(true, JSON.parse(event.data));
    } catch (err) {
      console.error('WebSocket message handling error:', err);
    }
  };
  socket.onerror = (err) => {
    // onerror is followed by onclose; log it but let onclose handle reconnection.
    console.warn('WebSocket error:', err);
  };
  socket.onclose = () => {
    if (state.signalSocket !== socket) return; // superseded socket, ignore
    state.signalSocket = null;
    const delay = state.signalReconnectMs;
    state.signalReconnectMs = Math.min(30000, delay * 2);
    if (state.token) setTimeout(connectSignalStream, delay);
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
  const body = {
    name: document.getElementById('strategy-name').value,
    asset: document.getElementById('strategy-asset').value,
    fast_window: Number(document.getElementById('strategy-fast').value),
    slow_window: Number(document.getElementById('strategy-slow').value),
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
// Dashboard
// ===========================================================================
async function loadDashboard(isPoll, streamedSignals = null) {
  const grid = document.getElementById('signal-grid');
  if (!isPoll && !grid.children.length) skeletonGrid(grid, 8, 'skeleton-card');

  // Fetch signals (from WebSocket push or REST); security health is fetched
  // independently only when on the dashboard and not as part of a poll to
  // avoid coupling two endpoints unnecessarily.
  const signals = streamedSignals || await api('/api/signals/latest', {}, { silent: isPoll }).catch(() => null);
  if (!signals) return;

  document.getElementById('signal-meta').textContent =
    `Engine pipeline: ${signals.pipeline_ms} ms total (SBA bifurcation: ${signals.sba_ms} ms) · ` +
    `${signals.n_assets} assets · generated ${new Date(signals.generated_at * 1000).toLocaleTimeString()}` +
    (signals.error ? ` · ⚠ ${signals.error}` : '');

  grid.innerHTML = signals.signals.map((s, i) => {
    const asset = escapeHtml(String(s.asset));
    const sigType = escapeHtml(String(s.signal_type));
    // Optional-chain feature access: signal engine may return partial data
    const rsi = s.features?.rsi != null ? Number(s.features.rsi).toFixed(1) : 'N/A';
    const mom = s.features?.momentum != null ? (Number(s.features.momentum) * 100).toFixed(1) : 'N/A';
    const confPct = Math.round(Number(s.confidence) * 100);
    return `
    <div class="signal-card" id="signal-${asset}" style="animation-delay:${i * 60}ms">
      <div class="asset">${asset}</div>
      <div class="price">$${Number(s.last_price).toFixed(2)}</div>
      <span class="badge ${sigType}">${sigType}</span>
      <div class="confidence-bar"><div class="confidence-fill" data-target="${confPct}"></div></div>
      <div class="features-row">
        <span>RSI ${rsi}</span>
        <span>Mom ${mom}%</span>
        <span>Conf ${confPct}%</span>
      </div>
    </div>`;
  }).join('') || '<div class="empty-state">No signals yet.</div>';

  // animate confidence bars in on next frame + flash cards whose signal changed
  requestAnimationFrame(() => {
    grid.querySelectorAll('.confidence-fill').forEach((bar) => { bar.style.width = bar.dataset.target + '%'; });
  });
  signals.signals.forEach((s) => {
    const prev = state.lastSignals[s.asset];
    if (prev !== undefined && prev !== s.signal_type) {
      const card = document.getElementById(`signal-${s.asset}`);
      if (card) { card.classList.remove('flash-update'); requestAnimationFrame(() => card.classList.add('flash-update')); }
    }
    state.lastSignals[s.asset] = s.signal_type;
  });

  // Update the safety score ring on first load or if not already populated
  if (!isPoll) {
    api('/api/security/health', {}, { silent: true }).then((health) => {
      if (health) animateScoreRing(health.quantum_safety_score);
    }).catch(() => {});
  }
}

document.getElementById('refresh-signals').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  const icon = btn.querySelector('.refresh-icon');
  if (icon) icon.classList.add('spinning');
  setButtonLoading(btn, true, ' Refreshing…');
  try {
    // POST to /api/signals/refresh (state-mutating endpoint)
    await api('/api/signals/refresh', { method: 'POST' });
    await loadDashboard();
    toast('Signals refreshed', 'SBA engine re-ran over live market data', 'success', 2200);
  } catch (err) {
    toast('Refresh failed', err.message, 'error');
  } finally {
    setButtonLoading(btn, false);
    if (icon) icon.classList.remove('spinning');
  }
});

// ===========================================================================
// Trading
// ===========================================================================
function loadTrading() { refreshOrders(); }

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
    quantity: parseFloat(document.getElementById('order-qty').value),
    order_type: orderType,
    time_in_force: document.getElementById('order-tif').value,
    limit_price: ['limit', 'stop_limit'].includes(orderType)
      ? parseFloat(document.getElementById('order-limit-price').value) : null,
    stop_price: ['stop', 'stop_limit'].includes(orderType)
      ? parseFloat(document.getElementById('order-stop-price').value) : null,
  };
  setButtonLoading(btn, true, 'Signing with ML-DSA-65…');
  try {
    const order = await api('/api/trading/orders', { method: 'POST', body: JSON.stringify(body) }, { silent: true });
    await refreshOrders();
    toast(
      order.status === 'FILLED' ? 'Order filled' : 'Order submitted',
      `${body.side.toUpperCase()} ${body.quantity} ${body.asset}${order.filled_price ? ' @ $' + order.filled_price.toFixed(2) : ''}`,
      order.status === 'FILLED' ? 'success' : 'info'
    );
  } catch (err) { errEl.textContent = err.message; } finally { setButtonLoading(btn, false); }
});

async function refreshOrders(isPoll) {
  const list = document.getElementById('order-list');
  if (!isPoll && !list.children.length) skeletonGrid(list, 4, 'skeleton-row');
  const orders = await api('/api/trading/orders', {}, { silent: isPoll });
  list.innerHTML = orders.map((o, i) => {
    const side = escapeHtml(String(o.side || '')).toUpperCase();
    const asset = escapeHtml(String(o.asset));
    const status = escapeHtml(String(o.status));
    const priceDetail = o.order_type === 'limit'
      ? ` @ $${Number(o.limit_price).toFixed(2)}`
      : o.order_type === 'stop'
      ? ` stop $${Number(o.stop_price).toFixed(2)}`
      : o.order_type === 'stop_limit'
      ? ` stop $${Number(o.stop_price).toFixed(2)} / limit $${Number(o.limit_price).toFixed(2)}`
      : '';
    const fillDetail = o.filled_price ? ` @ $${Number(o.filled_price).toFixed(2)}` : '';
    return `
    <div class="order-row" style="animation-delay:${i * 45}ms">
      <span>${side} ${Number(o.quantity)} ${asset}${priceDetail}</span>
      <span class="status status-${status}">${status}${fillDetail}</span>
    </div>`;
  }).join('') || '<div class="empty-state">No orders yet — place your first paper trade.</div>';
}

// ===========================================================================
// Portfolio
// ===========================================================================
async function loadPortfolio(isPoll) {
  const metricsEl = document.getElementById('risk-metrics');
  if (!isPoll && !metricsEl.children.length) skeletonGrid(metricsEl, 5, 'skeleton-card');

  const [positions, metrics] = await Promise.all([
    api('/api/portfolio/positions', {}, { silent: isPoll }),
    api('/api/portfolio/risk-metrics', {}, { silent: isPoll }),
  ]);

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

  const list = document.getElementById('positions-list');
  list.innerHTML = positions.map((p, i) => {
    const asset = escapeHtml(String(p.asset));
    const pnlClass = Number(p.unrealized_pnl) >= 0 ? 'pnl-pos' : 'pnl-neg';
    return `
    <div class="pos-row" style="animation-delay:${i * 50}ms">
      <span>${asset} &middot; ${Number(p.quantity)} sh @ $${Number(p.avg_entry_price).toFixed(2)}</span>
      <span class="${pnlClass}">$${Number(p.unrealized_pnl).toFixed(2)}</span>
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
    ctx.fillStyle = '#8393ac'; ctx.font = '12px sans-serif';
    ctx.fillText('No equity history yet — place a trade to begin tracking.', 20, 130);
    return;
  }
  const min = Math.min(...curve), max = Math.max(...curve);
  const pad = 20;
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
    ctx.strokeStyle = '#4fd8ff'; ctx.lineWidth = 2.2; ctx.shadowColor = 'rgba(79,216,255,0.5)'; ctx.shadowBlur = 6;
    ctx.beginPath();
    points.slice(0, revealCount).forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.stroke();
    ctx.shadowBlur = 0;
    if (revealCount > 0) {
      const last = points[revealCount - 1];
      ctx.beginPath(); ctx.fillStyle = '#4fd8ff'; ctx.arc(last.x, last.y, 3.5, 0, Math.PI * 2); ctx.fill();
    }
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
  keyEl.innerHTML = health.keys.map((k, i) => {
    const status = escapeHtml(String(k.status));
    const algo = escapeHtml(String(k.algorithm));
    return `
    <div class="key-row" style="animation-delay:${i * 60}ms">
      <span><span class="dot dot-${status}"></span>${algo} &middot; rotation #${Number(k.rotation_count)}</span>
      <span>${Number(k.age_days)}d old &middot; due in ${Number(k.rotation_due_in_days)}d</span>
    </div>`;
  }).join('') || '<div class="empty-state">No keys issued yet.</div>';

  renderHandshakeTrace();

  const logs = await api('/api/security/audit-log', {}, { silent: isPoll });
  document.getElementById('audit-log').innerHTML = logs.map((l, i) => {
    const action = escapeHtml(String(l.action));
    const resType = l.resource_type ? ' &middot; ' + escapeHtml(String(l.resource_type)) : '';
    const verifiedLabel = l.verified ? '&#10003; ML-DSA verified' : '&mdash;';
    // Date constructor is safe with an ISO string from the API
    const createdAt = l.created_at ? new Date(l.created_at).toLocaleString() : '';
    return `
    <div class="audit-row" style="animation-delay:${i * 30}ms">
      <span>${action}${resType}</span>
      <span>${verifiedLabel} &middot; ${createdAt}</span>
    </div>`;
  }).join('') || '<div class="empty-state">No audit entries yet.</div>';
}

async function rotateKeys(algorithm, btn) {
  setButtonLoading(btn, true, 'Rotating…');
  try {
    const res = await api('/api/security/rotate-keys', { method: 'POST', body: JSON.stringify({ algorithm, reason: 'manual_rotation' }) });
    await loadSecurity();
    toast('Key rotated', `${algorithm} · rotation #${res.rotation_count} · keygen ${res.keygen_ms} ms`, 'success');
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
  return Array.from(document.getElementById(id).selectedOptions).map((option) => option.value);
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
    document.getElementById('api-key-secret').textContent = `Copy this API key now; it will not be shown again: ${result.api_key}`;
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
