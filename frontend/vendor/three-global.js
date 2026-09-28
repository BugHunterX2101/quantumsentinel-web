// Exposes the self-hosted Three.js ES module build as the global `THREE`
// that bg3d.js expects. Three.js r160+ no longer ships a UMD build, and
// loading it same-origin keeps CSP at script-src 'self'.
// three.module.min.js: three@0.161.0 from the npm registry, tarball verified
// against its published sha512 integrity (MIT licence: three.LICENSE.txt).
import * as THREE from './three.module.min.js';

window.THREE = THREE;
