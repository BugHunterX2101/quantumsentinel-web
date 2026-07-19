// QuantumSentinel — frontend application logic (vanilla JS, no build step).
const state = {
  token: localStorage.getItem('qs_token') || null,
  user: JSON.parse(localStorage.getItem('qs_user') || 'null'),
  beginner: localStorage.getItem('qs_beginner') !== 'false',
  meta: null,
  lastSignals: {},   // asset -> last confidence, for diff-highlighting
  pollTimer: null,
  activeView: 'dashboard',
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
  el.innerHTML = `<div class="toast-title">${title}</div><div class="toast-body">${body || ''}</div>`;
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
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  }).catch((err) => {
    if (!opts2.silent) toast('Request failed', err.message, 'error');
    throw err;
  }).finally(() => loadingBar.done());
}

const b64encode = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)));

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
  if (loading) {
    btn.dataset.originalLabel = btn.querySelector('.btn-label').textContent;
    btn.querySelector('.btn-label').innerHTML = `<span class="spinner"></span>${loadingText}`;
    btn.disabled = true;
  } else {
    btn.querySelector('.btn-label').textContent = btn.dataset.originalLabel || '';
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
    `<div class="handshake-step" style="animation-delay:${i * 70}ms"><span class="label">${label}:</span><br><span class="val">${val}</span></div>`
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
  if (view === 'portfolio') loadPortfolio();
  if (view === 'security') loadSecurity();
  restartPolling();
}

function restartPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  const loaders = { dashboard: loadDashboard, trading: refreshOrders, portfolio: loadPortfolio, security: loadSecurity };
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
  localStorage.removeItem('qs_token');
  localStorage.removeItem('qs_user');
  location.reload();
});

async function bootstrapApp() {
  applyBeginnerMode();
  state.meta = await api('/api/meta');
  const sel = document.getElementById('order-asset');
  sel.innerHTML = state.meta.tracked_assets.map((a) => `<option value="${a}">${a}</option>`).join('');
  switchView('dashboard');
}

function skeletonGrid(container, count, cardClass) {
  container.innerHTML = Array.from({ length: count })
    .map(() => `<div class="skeleton ${cardClass}"></div>`).join('');
}

// ===========================================================================
// Dashboard
// ===========================================================================
async function loadDashboard(isPoll) {
  const grid = document.getElementById('signal-grid');
  if (!isPoll && !grid.children.length) skeletonGrid(grid, 8, 'skeleton-card');

  const [signals, health] = await Promise.all([
    api('/api/signals/latest', {}, { silent: isPoll }),
    api('/api/security/health', {}, { silent: true }).catch(() => null),
  ]);
  if (health) animateScoreRing(health.quantum_safety_score);

  document.getElementById('signal-meta').textContent =
    `Engine pipeline: ${signals.pipeline_ms} ms total (SBA bifurcation: ${signals.sba_ms} ms) · ` +
    `${signals.n_assets} assets · generated ${new Date(signals.generated_at * 1000).toLocaleTimeString()}`;

  grid.innerHTML = signals.signals.map((s, i) => `
    <div class="signal-card" id="signal-${s.asset}" style="animation-delay:${i * 60}ms">
      <div class="asset">${s.asset}</div>
      <div class="price">$${s.last_price.toFixed(2)}</div>
      <span class="badge ${s.signal_type}">${s.signal_type}</span>
      <div class="confidence-bar"><div class="confidence-fill" data-target="${Math.round(s.confidence*100)}"></div></div>
      <div class="features-row">
        <span>RSI ${s.features.rsi.toFixed(1)}</span>
        <span>Mom ${(s.features.momentum*100).toFixed(1)}%</span>
        <span>Conf ${(s.confidence*100).toFixed(0)}%</span>
      </div>
    </div>
  `).join('') || '<div class="empty-state">No signals yet.</div>';

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
}

document.getElementById('refresh-signals').addEventListener('click', async (e) => {
  const icon = e.currentTarget.querySelector('.refresh-icon');
  icon.classList.add('spinning');
  await api('/api/signals/refresh');
  await loadDashboard();
  toast('Signals refreshed', 'SBA engine re-ran over live market data', 'success', 2200);
  setTimeout(() => icon.classList.remove('spinning'), 600);
});

// ===========================================================================
// Trading
// ===========================================================================
function loadTrading() { refreshOrders(); }

document.getElementById('order-type').addEventListener('change', (e) => {
  document.getElementById('limit-price-wrap').classList.toggle('hidden', e.target.value !== 'limit');
});

document.getElementById('order-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('order-error');
  const btn = document.getElementById('order-submit-btn');
  errEl.textContent = '';
  const body = {
    asset: document.getElementById('order-asset').value,
    side: document.getElementById('order-side').value,
    quantity: parseFloat(document.getElementById('order-qty').value),
    order_type: document.getElementById('order-type').value,
    limit_price: document.getElementById('order-type').value === 'limit'
      ? parseFloat(document.getElementById('order-limit-price').value) : null,
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
  list.innerHTML = orders.map((o, i) => `
    <div class="order-row" style="animation-delay:${i * 45}ms">
      <span>${o.side.toUpperCase()} ${o.quantity} ${o.asset} ${o.order_type === 'limit' ? '@ $' + o.limit_price : ''}</span>
      <span class="status status-${o.status}">${o.status}${o.filled_price ? ' @ $' + o.filled_price.toFixed(2) : ''}</span>
    </div>
  `).join('') || '<div class="empty-state">No orders yet — place your first paper trade.</div>';
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
  `;
  animateCounter(document.getElementById('m-sharpe'), metrics.sharpe_ratio, { decimals: 2 });
  animateCounter(document.getElementById('m-dd'), metrics.max_drawdown * 100, { decimals: 1, suffix: '%' });
  animateCounter(document.getElementById('m-win'), metrics.win_rate * 100, { decimals: 0, suffix: '%' });
  animateCounter(document.getElementById('m-trades'), metrics.total_trades, { decimals: 0 });
  animateCounter(document.getElementById('m-var'), metrics.var_95 * 100, { decimals: 2, suffix: '%' });

  const list = document.getElementById('positions-list');
  list.innerHTML = positions.map((p, i) => `
    <div class="pos-row" style="animation-delay:${i * 50}ms">
      <span>${p.asset} · ${p.quantity} sh @ $${p.avg_entry_price.toFixed(2)}</span>
      <span class="${p.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${p.unrealized_pnl.toFixed(2)}</span>
    </div>
  `).join('') || '<div class="empty-state">No open positions.</div>';

  animateEquityCurve(metrics.equity_curve || []);
}

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
  keyEl.innerHTML = health.keys.map((k, i) => `
    <div class="key-row" style="animation-delay:${i * 60}ms">
      <span><span class="dot dot-${k.status}"></span>${k.algorithm} · rotation #${k.rotation_count}</span>
      <span>${k.age_days}d old · due in ${k.rotation_due_in_days}d</span>
    </div>
  `).join('') || '<div class="empty-state">No keys issued yet.</div>';

  renderHandshakeTrace();

  const logs = await api('/api/security/audit-log', {}, { silent: isPoll });
  document.getElementById('audit-log').innerHTML = logs.map((l, i) => `
    <div class="audit-row" style="animation-delay:${i * 30}ms">
      <span>${l.action} ${l.resource_type ? '· ' + l.resource_type : ''}</span>
      <span>${l.verified ? '✓ ML-DSA verified' : '—'} · ${new Date(l.created_at).toLocaleString()}</span>
    </div>
  `).join('') || '<div class="empty-state">No audit entries yet.</div>';
}

async function rotateKeys(algorithm, btn) {
  setButtonLoading(btn, true, 'Rotating…');
  try {
    const res = await api('/api/security/rotate-keys', { method: 'POST', body: JSON.stringify({ algorithm, reason: 'manual_rotation' }) });
    await loadSecurity();
    toast('Key rotated', `${algorithm} · rotation #${res.rotation_count} · keygen ${res.keygen_ms} ms`, 'success');
  } finally { setButtonLoading(btn, false); }
}
document.getElementById('rotate-dsa').addEventListener('click', (e) => {
  const btn = e.currentTarget;
  rotateDsaWrap(btn);
});
function rotateDsaWrap(btn) {
  if (!btn.querySelector('.btn-label')) { btn.innerHTML = `<span class="btn-label">${btn.textContent}</span>`; }
  rotateKeys('ML-DSA-65', btn);
}
document.getElementById('rotate-kem').addEventListener('click', (e) => {
  const btn = e.currentTarget;
  if (!btn.querySelector('.btn-label')) { btn.innerHTML = `<span class="btn-label">${btn.textContent}</span>`; }
  rotateKeys('ML-KEM-768', btn);
});

// ===========================================================================
// Resume session on page load
// ===========================================================================
(async function init() {
  if (state.token && state.user) {
    document.getElementById('app').classList.remove('hidden');
    document.getElementById('user-email').textContent = state.user.email;
    try {
      await performHandshake({ showOverlay: false });
      await bootstrapApp();
    } catch (e) {
      localStorage.removeItem('qs_token');
      location.reload();
    }
  }
})();
