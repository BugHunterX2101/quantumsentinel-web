/**
 * QuantumSentinel — 3D WebGL Background Engine
 * Uses Three.js for GPU-accelerated particle networks on all pages.
 * Each page gets its own colour palette and motion profile.
 */
(function () {
  'use strict';

  // ─── Config per-page-section ───────────────────────────────────────────────
  const CONFIGS = {
    auth: {
      particleCount: 180,
      color1: 0x1842A8,   // royal blue
      color2: 0x3B68E8,   // mid blue
      lineColor: 0xC5D0F0,
      lineOpacity: 0.18,
      particleSize: 1.6,
      speed: 0.00025,
      depth: 60,
      connectDist: 22,
    },
    dashboard: {
      particleCount: 220,
      color1: 0x1842A8,
      color2: 0x16A34A,   // green accent for trading
      lineColor: 0xD8E0F0,
      lineOpacity: 0.12,
      particleSize: 1.4,
      speed: 0.00018,
      depth: 50,
      connectDist: 20,
    },
    trading: {
      particleCount: 160,
      color1: 0x1842A8,
      color2: 0xF59E0B,   // amber for order signals
      lineColor: 0xE5D8A0,
      lineOpacity: 0.12,
      particleSize: 1.5,
      speed: 0.0002,
      depth: 45,
      connectDist: 18,
    },
    default: {
      particleCount: 150,
      color1: 0x1842A8,
      color2: 0x5B86E5,
      lineColor: 0xD0D8F0,
      lineOpacity: 0.14,
      particleSize: 1.4,
      speed: 0.0002,
      depth: 50,
      connectDist: 20,
    },
  };

  // ─── Scene instances per canvas ────────────────────────────────────────────
  const instances = new Map();

  function lerp(a, b, t) { return a + (b - a) * t; }

  function createScene(canvas, configKey) {
    const cfg = CONFIGS[configKey] || CONFIGS.default;
    const { THREE } = window;
    if (!THREE || !canvas) return;

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 500);
    camera.position.z = 80;

    // ── Particles ─────────────────────────────────────────────────────────────
    const positions = new Float32Array(cfg.particleCount * 3);
    const velocities = new Float32Array(cfg.particleCount * 3);
    const colors = new Float32Array(cfg.particleCount * 3);

    const c1 = new THREE.Color(cfg.color1);
    const c2 = new THREE.Color(cfg.color2);

    for (let i = 0; i < cfg.particleCount; i++) {
      const i3 = i * 3;
      positions[i3]     = (Math.random() - 0.5) * 160;
      positions[i3 + 1] = (Math.random() - 0.5) * 100;
      positions[i3 + 2] = (Math.random() - 0.5) * cfg.depth;
      velocities[i3]     = (Math.random() - 0.5) * cfg.speed * 2;
      velocities[i3 + 1] = (Math.random() - 0.5) * cfg.speed * 2;
      velocities[i3 + 2] = (Math.random() - 0.5) * cfg.speed;
      const t = Math.random();
      const c = new THREE.Color().lerpColors(c1, c2, t);
      colors[i3] = c.r; colors[i3 + 1] = c.g; colors[i3 + 2] = c.b;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
      size: cfg.particleSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(geo, mat);
    scene.add(points);

    // ── Connection lines ───────────────────────────────────────────────────────
    // Pre-allocate max possible pairs; update dynamically each frame
    const MAX_LINES = cfg.particleCount * 6;
    const linePositions = new Float32Array(MAX_LINES * 6);
    const lineColors = new Float32Array(MAX_LINES * 6);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
    lineGeo.setDrawRange(0, 0);
    const lineMat = new THREE.LineSegmentsMaterial({
      vertexColors: true,
      transparent: true,
      opacity: cfg.lineOpacity,
    });
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lines);

    const lineCol = new THREE.Color(cfg.lineColor);
    let frameId;
    let lastResize = 0;

    function resize() {
      const w = canvas.clientWidth || canvas.offsetWidth || window.innerWidth;
      const h = canvas.clientHeight || canvas.offsetHeight || window.innerHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    resize();

    // Mouse parallax
    let mx = 0, my = 0;
    canvas.addEventListener('mousemove', (e) => {
      mx = (e.clientX / window.innerWidth - 0.5) * 2;
      my = -(e.clientY / window.innerHeight - 0.5) * 2;
    }, { passive: true });

    let clock = 0;

    function animate() {
      frameId = requestAnimationFrame(animate);
      clock += cfg.speed * 60;

      // Resize if needed (throttled to 500ms)
      const now = performance.now();
      if (now - lastResize > 500) {
        resize();
        lastResize = now;
      }

      // Gentle camera drift + parallax
      camera.position.x = lerp(camera.position.x, mx * 8, 0.02);
      camera.position.y = lerp(camera.position.y, my * 5, 0.02);

      // Move particles
      const pos = geo.attributes.position.array;
      for (let i = 0; i < cfg.particleCount; i++) {
        const i3 = i * 3;
        pos[i3]     += velocities[i3];
        pos[i3 + 1] += velocities[i3 + 1];
        pos[i3 + 2] += velocities[i3 + 2];
        // Wrap around bounds
        if (Math.abs(pos[i3])     > 85)  velocities[i3]     *= -1;
        if (Math.abs(pos[i3 + 1]) > 55)  velocities[i3 + 1] *= -1;
        if (Math.abs(pos[i3 + 2]) > cfg.depth / 2) velocities[i3 + 2] *= -1;
        // Subtle oscillation
        pos[i3 + 1] += Math.sin(clock + i * 0.5) * 0.008;
      }
      geo.attributes.position.needsUpdate = true;

      // Dynamic line connections
      const lp = lineGeo.attributes.position.array;
      const lc = lineGeo.attributes.color.array;
      let lineIdx = 0;
      const dist2 = cfg.connectDist * cfg.connectDist;

      for (let i = 0; i < cfg.particleCount && lineIdx < MAX_LINES * 3; i++) {
        const ix = pos[i * 3], iy = pos[i * 3 + 1], iz = pos[i * 3 + 2];
        for (let j = i + 1; j < cfg.particleCount && lineIdx < MAX_LINES * 3; j++) {
          const dx = ix - pos[j * 3];
          const dy = iy - pos[j * 3 + 1];
          const dz = iz - pos[j * 3 + 2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < dist2) {
            const alpha = 1.0 - d2 / dist2;
            const li = lineIdx / 3 * 6;
            // from
            lp[li]     = ix; lp[li + 1] = iy; lp[li + 2] = iz;
            // to
            lp[li + 3] = pos[j * 3]; lp[li + 4] = pos[j * 3 + 1]; lp[li + 5] = pos[j * 3 + 2];
            // colors (alpha encoded in rgb intensity)
            lc[li]     = lineCol.r * alpha; lc[li + 1] = lineCol.g * alpha; lc[li + 2] = lineCol.b * alpha;
            lc[li + 3] = lineCol.r * alpha; lc[li + 4] = lineCol.g * alpha; lc[li + 5] = lineCol.b * alpha;
            lineIdx += 2;  // 2 vertices per line segment
          }
        }
      }
      lineGeo.setDrawRange(0, lineIdx);
      lineGeo.attributes.position.needsUpdate = true;
      lineGeo.attributes.color.needsUpdate = true;

      // Slow global rotation for depth
      scene.rotation.y = Math.sin(clock * 0.3) * 0.04;
      scene.rotation.x = Math.cos(clock * 0.2) * 0.02;

      renderer.render(scene, camera);
    }

    animate();

    return {
      destroy() {
        cancelAnimationFrame(frameId);
        renderer.dispose();
        geo.dispose(); mat.dispose();
        lineGeo.dispose(); lineMat.dispose();
      },
    };
  }

  // ─── Public API ────────────────────────────────────────────────────────────
  window.QS3D = {
    init(canvasId, configKey) {
      if (instances.has(canvasId)) return;
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      // Only initialise if Three.js is loaded
      if (!window.THREE) {
        window.addEventListener('three-ready', () => this.init(canvasId, configKey), { once: true });
        return;
      }
      const inst = createScene(canvas, configKey);
      if (inst) instances.set(canvasId, inst);
    },
    destroy(canvasId) {
      const inst = instances.get(canvasId);
      if (inst) { inst.destroy(); instances.delete(canvasId); }
    },
    destroyAll() {
      instances.forEach(inst => inst.destroy());
      instances.clear();
    },
  };
})();
