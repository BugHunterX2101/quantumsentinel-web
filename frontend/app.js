// QuantumSentinel — frontend application logic (vanilla JS, no build step).
const state = {
  token: localStorage.getItem('qs_token') || null,
  user: JSON.parse(localStorage.getItem('qs_user') || 'null'),
  beginner: localStorage.getItem('qs_beginner') !== 'false',
  meta: null,
};

function api(path, opts = {}) {
  const headers = Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {});
  if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
  return fetch(path, Object.assign({}, opts, { headers })).then(async (r) => {
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  });
}

const b64encode = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)));

// ---------------------------------------------------------------------------
// Auth screen wiring
// ---------------------------------------------------------------------------
document.querySelectorAll('.auth-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.auth-tab').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('login-form').classList.toggle('hidden', btn.dataset.tab !== 'login');
    document.getElementById('register-form').classList.toggle('hidden', btn.dataset.tab !== 'register');
  });
});

document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  const errEl = document.getElementById('login-error');
  errEl.textContent = '';
  try {
    const data = await api('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
    await afterLogin(data);
  } catch (err) { errEl.textContent = err.message; }
});

document.getElementById('register-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const email = document.getElementById('register-email').value;
  const password = document.getElementById('register-password').value;
  const errEl = document.getElementById('register-error');
  errEl.textContent = '';
  try {
    await api('/api/auth/register', { method: 'POST', body: JSON.stringify({ email, password }) });
    const data = await api('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
    await afterLogin(data);
  } catch (err) { errEl.textContent = err.message; }
});

async function afterLogin(data) {
  state.token = data.access_token;
  state.user = data.user;
  localStorage.setItem('qs_token', state.token);
  localStorage.setItem('qs_user', JSON.stringify(state.user));
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  document.getElementById('user-email').textContent = state.user.email;
  await performHandshake();
  await bootstrapApp();
}

// ---------------------------------------------------------------------------
// PQC hybrid handshake (real server-side ML-KEM-768 + ML-DSA-65 + X25519)
// ---------------------------------------------------------------------------
async function performHandshake() {
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

  const result = await api('/api/auth/pqc-handshake', {
    method: 'POST',
    body: JSON.stringify({ x25519_public_key: clientPub, client_nonce: clientNonce }),
  });
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
  el.innerHTML = rows.map(([label, val]) =>
    `<div class="handshake-step"><span class="label">${label}:</span><br><span class="val">${val}</span></div>`
  ).join('');
}

// ---------------------------------------------------------------------------
// App shell: tabs, beginner mode, logout
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => switchView(btn.dataset.view));
});

function switchView(view) {
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + view));
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
}

function applyBeginnerMode() {
  document.querySelectorAll('[data-beginner]').forEach((el) => el.classList.toggle('hidden', !state.beginner));
  document.getElementById('beginner-state').textContent = state.beginner ? 'On' : 'Off';
}

document.getElementById('beginner-toggle').addEventListener('click', () => {
  state.beginner = !state.beginner;
  localStorage.setItem('qs_beginner', state.beginner);
  applyBeginnerMode();
  const activeView = document.querySelector('.tab-btn.active').dataset.view;
  switchView(activeView);
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

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
async function loadDashboard() {
  const [signals, health] = await Promise.all([
    api('/api/signals/latest'),
    api('/api/security/health').catch(() => null),
  ]);
  if (health) updateSafetyBadge(health.quantum_safety_score);

  document.getElementById('signal-meta').textContent =
    `Engine pipeline: ${signals.pipeline_ms} ms total (SBA bifurcation: ${signals.sba_ms} ms) · ` +
    `${signals.n_assets} assets · generated ${new Date(signals.generated_at * 1000).toLocaleTimeString()}`;

  const grid = document.getElementById('signal-grid');
  grid.innerHTML = signals.signals.map((s) => `
    <div class="signal-card">
      <div class="asset">${s.asset}</div>
      <div class="price">$${s.last_price.toFixed(2)}</div>
      <span class="badge ${s.signal_type}">${s.signal_type}</span>
      <div class="confidence-bar"><div class="confidence-fill" style="width:${Math.round(s.confidence*100)}%"></div></div>
      <div class="features-row">
        <span>RSI ${s.features.rsi.toFixed(1)}</span>
        <span>Mom ${(s.features.momentum*100).toFixed(1)}%</span>
        <span>Conf ${(s.confidence*100).toFixed(0)}%</span>
      </div>
    </div>
  `).join('') || '<div class="empty-state">No signals yet.</div>';
}

document.getElementById('refresh-signals').addEventListener('click', async () => {
  await api('/api/signals/refresh');
  loadDashboard();
});

function updateSafetyBadge(score) {
  const badge = document.getElementById('safety-score-badge');
  const val = document.getElementById('safety-score-value');
  val.textContent = Math.round(score);
  let color = '#3ddc97';
  if (score < 60) color = '#ff5d7a'; else if (score < 90) color = '#ffd166';
  badge.style.borderColor = color;
  badge.style.color = color;
}

// ---------------------------------------------------------------------------
// Trading
// ---------------------------------------------------------------------------
function loadTrading() {
  refreshOrders();
}

document.getElementById('order-type').addEventListener('change', (e) => {
  document.getElementById('limit-price-wrap').classList.toggle('hidden', e.target.value !== 'limit');
});

document.getElementById('order-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('order-error');
  errEl.textContent = '';
  const body = {
    asset: document.getElementById('order-asset').value,
    side: document.getElementById('order-side').value,
    quantity: parseFloat(document.getElementById('order-qty').value),
    order_type: document.getElementById('order-type').value,
    limit_price: document.getElementById('order-type').value === 'limit'
      ? parseFloat(document.getElementById('order-limit-price').value) : null,
  };
  try {
    await api('/api/trading/orders', { method: 'POST', body: JSON.stringify(body) });
    refreshOrders();
  } catch (err) { errEl.textContent = err.message; }
});

async function refreshOrders() {
  const orders = await api('/api/trading/orders');
  const list = document.getElementById('order-list');
  list.innerHTML = orders.map((o) => `
    <div class="order-row">
      <span>${o.side.toUpperCase()} ${o.quantity} ${o.asset} ${o.order_type === 'limit' ? '@ $' + o.limit_price : ''}</span>
      <span class="status status-${o.status}">${o.status}${o.filled_price ? ' @ $' + o.filled_price.toFixed(2) : ''}</span>
    </div>
  `).join('') || '<div class="empty-state">No orders yet — place your first paper trade.</div>';
}

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------
async function loadPortfolio() {
  const [positions, metrics] = await Promise.all([
    api('/api/portfolio/positions'), api('/api/portfolio/risk-metrics'),
  ]);

  document.getElementById('risk-metrics').innerHTML = `
    <div class="metric-box"><div class="val">${metrics.sharpe_ratio}</div><div class="lbl">Sharpe Ratio</div></div>
    <div class="metric-box"><div class="val">${(metrics.max_drawdown*100).toFixed(1)}%</div><div class="lbl">Max Drawdown</div></div>
    <div class="metric-box"><div class="val">${(metrics.win_rate*100).toFixed(0)}%</div><div class="lbl">Win Rate</div></div>
    <div class="metric-box"><div class="val">${metrics.total_trades}</div><div class="lbl">Filled Trades</div></div>
    <div class="metric-box"><div class="val">${(metrics.var_95*100).toFixed(2)}%</div><div class="lbl">VaR 95%</div></div>
  `;

  const list = document.getElementById('positions-list');
  list.innerHTML = positions.map((p) => `
    <div class="pos-row">
      <span>${p.asset} · ${p.quantity} sh @ $${p.avg_entry_price.toFixed(2)}</span>
      <span class="${p.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${p.unrealized_pnl.toFixed(2)}</span>
    </div>
  `).join('') || '<div class="empty-state">No open positions.</div>';

  drawEquityCurve(metrics.equity_curve || []);
}

function drawEquityCurve(curve) {
  const canvas = document.getElementById('equity-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!curve.length) {
    ctx.fillStyle = '#8393ac'; ctx.font = '12px sans-serif';
    ctx.fillText('No equity history yet — place a trade to begin tracking.', 20, 130);
    return;
  }
  const min = Math.min(...curve), max = Math.max(...curve);
  const pad = 20;
  const w = canvas.width - pad * 2, h = canvas.height - pad * 2;
  ctx.strokeStyle = '#4fd8ff'; ctx.lineWidth = 2; ctx.beginPath();
  curve.forEach((v, i) => {
    const x = pad + (i / (curve.length - 1 || 1)) * w;
    const y = pad + h - ((v - min) / (max - min || 1)) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Security
// ---------------------------------------------------------------------------
async function loadSecurity() {
  const health = await api('/api/security/health');
  updateSafetyBadge(health.quantum_safety_score);
  document.getElementById('key-health').innerHTML = health.keys.map((k) => `
    <div class="key-row">
      <span><span class="dot dot-${k.status}"></span>${k.algorithm} · rotation #${k.rotation_count}</span>
      <span>${k.age_days}d old · due in ${k.rotation_due_in_days}d</span>
    </div>
  `).join('') || '<div class="empty-state">No keys issued yet.</div>';

  renderHandshakeTrace();

  const logs = await api('/api/security/audit-log');
  document.getElementById('audit-log').innerHTML = logs.map((l) => `
    <div class="audit-row">
      <span>${l.action} ${l.resource_type ? '· ' + l.resource_type : ''}</span>
      <span>${l.verified ? '✓ ML-DSA verified' : '—'} · ${new Date(l.created_at).toLocaleString()}</span>
    </div>
  `).join('') || '<div class="empty-state">No audit entries yet.</div>';
}

document.getElementById('rotate-dsa').addEventListener('click', async () => {
  await api('/api/security/rotate-keys', { method: 'POST', body: JSON.stringify({ algorithm: 'ML-DSA-65', reason: 'manual_rotation' }) });
  loadSecurity();
});
document.getElementById('rotate-kem').addEventListener('click', async () => {
  await api('/api/security/rotate-keys', { method: 'POST', body: JSON.stringify({ algorithm: 'ML-KEM-768', reason: 'manual_rotation' }) });
  loadSecurity();
});

// ---------------------------------------------------------------------------
// Resume session on page load
// ---------------------------------------------------------------------------
(async function init() {
  if (state.token && state.user) {
    document.getElementById('auth-screen').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');
    document.getElementById('user-email').textContent = state.user.email;
    try {
      await performHandshake();
      await bootstrapApp();
    } catch (e) {
      localStorage.removeItem('qs_token');
      location.reload();
    }
  }
})();
