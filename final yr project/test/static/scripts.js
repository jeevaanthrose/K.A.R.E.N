'use strict';

/* ═══════════════════════════════════════════════════════
   BACKGROUND PARTICLE NETWORK
═══════════════════════════════════════════════════════ */
const bgCanvas = document.getElementById('canvas-bg');
const bgCtx    = bgCanvas.getContext('2d');
let   particles = [];

function resizeBg() {
  bgCanvas.width  = window.innerWidth;
  bgCanvas.height = window.innerHeight;
}

class Particle {
  constructor() {
    this.x     = Math.random() * bgCanvas.width;
    this.y     = Math.random() * bgCanvas.height;
    this.sz    = Math.random() * 1.4 + 0.4;
    this.vx    = (Math.random() - .5) * .7;
    this.vy    = (Math.random() - .5) * .7;
    this.phase = Math.random() * Math.PI * 2;
  }
  update() {
    this.x += this.vx; this.y += this.vy; this.phase += .03;
    if (this.x < 0 || this.x > bgCanvas.width)  this.vx *= -1;
    if (this.y < 0 || this.y > bgCanvas.height)  this.vy *= -1;
  }
  draw() {
    const r = this.sz + Math.sin(this.phase) * .35;
    bgCtx.beginPath();
    bgCtx.arc(this.x, this.y, r, 0, Math.PI * 2);
    bgCtx.fillStyle  = 'rgba(0,210,255,0.75)';
    bgCtx.shadowBlur = 7; bgCtx.shadowColor = '#00f3ff';
    bgCtx.fill(); bgCtx.shadowBlur = 0;
  }
}

function animateBg() {
  bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
  const cx = bgCanvas.width / 2, cy = bgCanvas.height / 2;
  const g  = bgCtx.createRadialGradient(cx, cy, 0, cx, cy, bgCanvas.width * .6);
  g.addColorStop(0, 'rgba(0,40,80,.10)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  bgCtx.fillStyle = g;
  bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);

  particles.forEach(p => { p.update(); p.draw(); });

  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x;
      const dy = particles[i].y - particles[j].y;
      const d  = Math.sqrt(dx * dx + dy * dy);
      if (d < 130) {
        bgCtx.beginPath();
        bgCtx.strokeStyle = `rgba(0,200,255,${(1 - d / 130) * .4})`;
        bgCtx.lineWidth   = .5;
        bgCtx.moveTo(particles[i].x, particles[i].y);
        bgCtx.lineTo(particles[j].x, particles[j].y);
        bgCtx.stroke();
      }
    }
  }
  requestAnimationFrame(animateBg);
}

/* ═══════════════════════════════════════════════════════
   TESSERACT  (4-D → 2-D, mathematically exact)
═══════════════════════════════════════════════════════ */
const tc   = document.getElementById('canvas-tesseract');
const tCtx = tc.getContext('2d');
const SIZE = 380;
tc.width = tc.height = SIZE;

const HS = 82, FOV = 500, ZCAM = 300, WFOV = 2.8 * HS;

function r4XW(v, a) { const [x,y,z,w]=v, c=Math.cos(a), s=Math.sin(a); return [x*c-w*s, y, z, x*s+w*c]; }
function r4YZ(v, a) { const [x,y,z,w]=v, c=Math.cos(a), s=Math.sin(a); return [x, y*c-z*s, y*s+z*c, w]; }
function r4XY(v, a) { const [x,y,z,w]=v, c=Math.cos(a), s=Math.sin(a); return [x*c-y*s, x*s+y*c, z, w]; }
function r4ZW(v, a) { const [x,y,z,w]=v, c=Math.cos(a), s=Math.sin(a); return [x, y, z*c-w*s, z*s+w*c]; }

function verts4D() {
  const v = [];
  for (let i = 0; i < 16; i++)
    v.push([(i&1?1:-1)*HS, (i&2?1:-1)*HS, (i&4?1:-1)*HS, (i&8?1:-1)*HS]);
  return v;
}
function p4to3(v) { const [x,y,z,w]=v, d=WFOV/(WFOV-w); return [x*d, y*d, z*d]; }
function rotX(v,  a) { const [x,y,z]=v, c=Math.cos(a), s=Math.sin(a); return [x, y*c-z*s, y*s+z*c]; }
function project(v) { const [x,y,z]=v, w=FOV/(FOV+z+ZCAM); return [SIZE/2+x*w, SIZE/2+y*w, w]; }

const EDGES = [];
for (let i = 0; i < 16; i++)
  for (let j = i+1; j < 16; j++) {
    const d = i ^ j;
    if (d && (d & (d-1)) === 0) EDGES.push([i, j]);
  }

function eType(i, j) {
  const iw = (i & 8) !== 0, jw = (j & 8) !== 0;
  if (iw && jw)  return 'inner';
  if (!iw && !jw) return 'outer';
  return 'conn';
}

function dEdge(p1, p2, col, alpha, lw, blur) {
  tCtx.beginPath();
  tCtx.moveTo(p1[0], p1[1]);
  tCtx.lineTo(p2[0], p2[1]);
  tCtx.strokeStyle = col.replace('$A', alpha);
  tCtx.lineWidth   = lw;
  tCtx.shadowBlur  = blur; tCtx.shadowColor = '#00f3ff';
  tCtx.stroke(); tCtx.shadowBlur = 0;
}
function dVert(p, alpha) {
  tCtx.beginPath();
  tCtx.arc(p[0], p[1], 2.2, 0, Math.PI * 2);
  tCtx.fillStyle  = `rgba(255,255,255,${alpha})`;
  tCtx.shadowBlur = 12; tCtx.shadowColor = '#00f3ff';
  tCtx.fill(); tCtx.shadowBlur = 0;
}

let orbP = 0;
function drawOrb() {
  orbP += .04;
  const r = 6 + Math.sin(orbP) * 4, cx = SIZE/2, cy = SIZE/2;
  const g = tCtx.createRadialGradient(cx, cy, 0, cx, cy, r * 4);
  g.addColorStop(0,  'rgba(255,255,255,.95)');
  g.addColorStop(.3, 'rgba(0,243,255,.9)');
  g.addColorStop(1,  'rgba(0,100,200,0)');
  tCtx.beginPath();
  tCtx.arc(cx, cy, r * 4, 0, Math.PI * 2);
  tCtx.fillStyle  = g;
  tCtx.shadowBlur = 28 + Math.sin(orbP) * 8;
  tCtx.shadowColor = '#00f3ff';
  tCtx.fill(); tCtx.shadowBlur = 0;
}

let tt = 0;
function drawTesseract() {
  tCtx.clearRect(0, 0, SIZE, SIZE);
  tt += .007;

  const raw = verts4D();
  const rot = raw.map(v => {
    v = r4XW(v, tt * 1.0); v = r4YZ(v, tt * .7);
    v = r4XY(v, tt * .4);  v = r4ZW(v, tt * .3);
    return v;
  });
  const p3 = rot.map(p4to3);
  const p2 = p3.map(v => project(rotX(v, .4)));

  const se = [...EDGES].sort((a, b) =>
    (p2[a[0]][2] + p2[a[1]][2]) - (p2[b[0]][2] + p2[b[1]][2])
  );
  tCtx.lineCap = 'round';

  se.forEach(([i,j]) => {
    if (eType(i,j) !== 'conn') return;
    const p1=p2[i], q=p2[j], dep=(p1[2]+q[2])/2;
    dEdge(p1, q, 'rgba(0,170,255,$A)', (0.2+dep*.5).toFixed(2), 1.0, 6);
  });
  se.forEach(([i,j]) => {
    if (eType(i,j) !== 'outer') return;
    const p1=p2[i], q=p2[j], dep=(p1[2]+q[2])/2;
    dEdge(p1, q, 'rgba(0,243,255,$A)', (0.45+dep*.55).toFixed(2), 1.8, 12);
  });
  se.forEach(([i,j]) => {
    if (eType(i,j) !== 'inner') return;
    const p1=p2[i], q=p2[j], dep=(p1[2]+q[2])/2;
    dEdge(p1, q, 'rgba(150,235,255,$A)', (0.55+dep*.45).toFixed(2), 1.4, 10);
  });
  p2.forEach(p => dVert(p, (0.35 + p[2] * .65).toFixed(2)));
  drawOrb();
  requestAnimationFrame(drawTesseract);
}

/* ═══════════════════════════════════════════════════════
   LOADING SEQUENCE
═══════════════════════════════════════════════════════ */
const bootLines = [
  'BIOS v9.1.3 ............. OK',
  'LOADING KERNEL MODULES ... OK',
  'ESTABLISHING UPLINK ...... OK',
  'DECRYPTING NODE KEYS ..... OK',
  'SATELLITE HANDSHAKE ...... OK',
  'INJECTING PAYLOAD ........ OK',
  'BYPASSING FIREWALL ....... OK',
  'PHOENIX CORE ONLINE ...... OK',
];

function runLoading() {
  const text    = document.getElementById('loading-text');
  const bar     = document.getElementById('progress-bar');
  const blog    = document.getElementById('boot-log');
  const phrases = ['INITIALIZING','LOADING KERNEL','SYNCING NODES','BYPASSING FIREWALL','CALIBRATING'];
  let p = 0, pi = 0, bi = 0;

  const bootIv = setInterval(() => {
    if (bi < bootLines.length) {
      const d = document.createElement('div');
      d.textContent = bootLines[bi++];
      blog.appendChild(d);
      blog.scrollTop = blog.scrollHeight;
    }
  }, 400);

  const iv = setInterval(() => {
    p += Math.random() * 2.4 + .5;
    const ni = Math.min(Math.floor(p / 22), phrases.length - 1);
    if (ni !== pi) {
      pi = ni; text.style.opacity = '0';
      setTimeout(() => { text.textContent = phrases[pi]; text.style.opacity = '1'; }, 240);
    }
    if (p >= 100) {
      p = 100; clearInterval(iv); clearInterval(bootIv);
      setTimeout(() => {
        text.style.opacity = '0';
        setTimeout(() => {
          text.textContent = 'ACCESS GRANTED';
          text.style.letterSpacing = '20px';
          text.style.opacity = '1';
          setTimeout(() => {
            text.style.opacity = '0';
            setTimeout(() => {
              text.textContent = 'WELCOME, PHOENIX';
              text.style.letterSpacing = '10px';
              text.style.opacity = '1';
              setTimeout(startDashboard, 2200);
            }, 300);
          }, 1800);
        }, 280);
      }, 400);
    }
    bar.style.width = p + '%';
  }, 120);
}

/* ═══════════════════════════════════════════════════════
   DASHBOARD INIT
═══════════════════════════════════════════════════════ */
function startDashboard() {
  document.getElementById('loading-screen').classList.add('hidden');
  document.getElementById('dashboard').classList.remove('hidden');
  initClock();
  initTerminalFeed();
  initRadar();
  initMap();
  initWave();
  initGauges();
  initOpsButtons();
  initPhxTerminal();
  setInterval(updateLiveStats, 3000);
}

/* ── Clock ── */
function initClock() {
  function tick() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-GB');
  }
  tick(); setInterval(tick, 1000);
}

/* ── Live feed ── */
const feedWords = [
  ['UPLINK','ESTABLISHED'],['ENCRYPT','COMPLETE'],['BYPASS','OK'],
  ['SIGNAL','ACQUIRED'],['PROXY','ROUTED'],['FIREWALL','EVADED'],
  ['PACKET','INTERCEPTED'],['TRACE','NULLIFIED'],['NODE','SYNCED'],
  ['BEACON','LOCKED'],['RELAY','ACTIVE'],['ROUTE','SPOOFED'],
];
function initTerminalFeed() {
  const log = document.getElementById('terminal-log');
  setInterval(() => {
    const [a, b] = feedWords[Math.floor(Math.random() * feedWords.length)];
    const hex    = '0x' + Math.random().toString(16).substr(2, 6).toUpperCase();
    const d      = document.createElement('div');
    d.textContent = `> ${a} ${hex} ... ${b}`;
    log.appendChild(d);
    while (log.children.length > 22) log.removeChild(log.firstChild);
  }, 600);
}

/* ═══════════════════════════════════════════════════════
   RADAR
═══════════════════════════════════════════════════════ */
function initRadar() {
  const c   = document.getElementById('radar-canvas');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height, cx = W/2, cy = H/2, R = cx - 4;
  let angle = 0;

  const blipData = [
    {r:.35,a:0.8},{r:.6,a:2.4},{r:.75,a:4.1},
    {r:.5, a:1.2},{r:.85,a:3.6},{r:.3, a:5.0},{r:.65,a:0.2}
  ];

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Background circle
    ctx.fillStyle = 'rgba(0,8,18,0.95)';
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI*2); ctx.fill();

    // Grid rings
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath(); ctx.arc(cx, cy, R*i/4, 0, Math.PI*2);
      ctx.strokeStyle = 'rgba(0,243,255,0.1)'; ctx.lineWidth = .8; ctx.stroke();
    }
    // Cross lines
    [0,1,2,3].forEach(i => {
      const rad = i * Math.PI / 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(rad)*R, cy + Math.sin(rad)*R);
      ctx.strokeStyle = 'rgba(0,243,255,0.08)'; ctx.lineWidth = .8; ctx.stroke();
    });

    // Sweep fill
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(angle);
    const gr = ctx.createLinearGradient(0, -R, 0, 0);
    gr.addColorStop(0, 'rgba(0,243,255,0)');
    gr.addColorStop(1, 'rgba(0,243,255,0.35)');
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, R, -Math.PI*.5, 0);
    ctx.fillStyle = gr; ctx.fill(); ctx.restore();

    // Sweep line
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(angle);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -R);
    ctx.strokeStyle = 'rgba(0,243,255,0.9)'; ctx.lineWidth = 1.5;
    ctx.shadowBlur = 8; ctx.shadowColor = '#00f3ff';
    ctx.stroke(); ctx.shadowBlur = 0; ctx.restore();

    // Blips
    blipData.forEach(b => {
      const diff = ((angle + Math.PI/2 - b.a) % (Math.PI*2) + Math.PI*2) % (Math.PI*2);
      const fade = Math.max(0, 1 - diff / Math.PI);
      if (fade > .05) {
        const bx = cx + Math.cos(b.a) * b.r * R;
        const by = cy + Math.sin(b.a) * b.r * R;
        ctx.beginPath(); ctx.arc(bx, by, 3, 0, Math.PI*2);
        ctx.fillStyle = `rgba(0,243,255,${fade})`;
        ctx.shadowBlur = 10; ctx.shadowColor = '#00f3ff';
        ctx.fill(); ctx.shadowBlur = 0;
      }
    });

    // Border
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI*2);
    ctx.strokeStyle = 'rgba(0,243,255,0.3)'; ctx.lineWidth = 1.5; ctx.stroke();

    angle += 0.03;
    if (angle > Math.PI * 2) angle -= Math.PI * 2;
    requestAnimationFrame(draw);
  }
  draw();
}

/* ═══════════════════════════════════════════════════════
   WORLD MAP  —  D3 + TopoJSON Natural Earth 110m
═══════════════════════════════════════════════════════ */
const MAP_NODES = [
  {lon:-74.0, lat:40.7,  label:'NEW YORK',    id:'NYC'},
  {lon:-0.1,  lat:51.5,  label:'LONDON',      id:'LON'},
  {lon:37.6,  lat:55.7,  label:'MOSCOW',      id:'MOS'},
  {lon:77.2,  lat:28.6,  label:'DELHI',       id:'DEL'},
  {lon:116.4, lat:39.9,  label:'BEIJING',     id:'BEJ'},
  {lon:139.7, lat:35.7,  label:'TOKYO',       id:'TOK'},
  {lon:-46.6, lat:-23.5, label:'SAO PAULO',   id:'SAO'},
  {lon:36.8,  lat:-1.3,  label:'NAIROBI',     id:'NAI'},
  {lon:151.2, lat:-33.9, label:'SYDNEY',      id:'SYD'},
  {lon:-118.2,lat:34.1,  label:'LOS ANGELES', id:'LAX'},
  {lon:2.3,   lat:48.9,  label:'PARIS',       id:'PAR'},
  {lon:55.3,  lat:25.2,  label:'DUBAI',       id:'DXB'},
];

const ARC_PAIRS = [
  [0,1],[1,2],[0,4],[2,5],[3,7],
  [6,0],[4,8],[9,6],[10,11],[11,3],
];

function initMap() {
  const wrap = document.getElementById('map-wrap');
  const svg  = d3.select('#map-svg');
  const tip  = document.getElementById('map-tooltip');
  let W, H, projection, path, arcT = 0;

  function resize() {
    W = wrap.clientWidth  || 600;
    H = wrap.clientHeight || 280;
    svg.attr('width', W).attr('height', H);
    projection = d3.geoNaturalEarth1()
      .scale(W / 6.2)
      .translate([W / 2, H / 2]);
    path = d3.geoPath().projection(projection);
    if (worldData) render();
  }

  let worldData = null;

  function render() {
    svg.selectAll('*').remove();

    // Ocean
    svg.append('rect').attr('width', W).attr('height', H).attr('fill', '#000e1c');

    // Graticule
    svg.append('path')
      .datum(d3.geoGraticule()())
      .attr('class', 'graticule')
      .attr('d', path);

    // Sphere outline
    svg.append('path')
      .datum({type:'Sphere'})
      .attr('d', path)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(0,243,255,0.15)')
      .attr('stroke-width', 0.8);

    // Land masses
    const land = topojson.feature(worldData, worldData.objects.countries);
    svg.append('g')
      .selectAll('path')
      .data(land.features)
      .enter().append('path')
      .attr('class', 'land')
      .attr('d', path);

    // Static dashed arc lines
    ARC_PAIRS.forEach(([fi, ti]) => {
      const n1 = MAP_NODES[fi], n2 = MAP_NODES[ti];
      svg.append('path')
        .datum({type:'LineString', coordinates:[[n1.lon,n1.lat],[n2.lon,n2.lat]]})
        .attr('class', 'arc-path')
        .attr('d', path);
    });

    // Node markers
    const nodeG = svg.append('g').attr('id', 'nodes-g');
    MAP_NODES.forEach(n => {
      const [px, py] = projection([n.lon, n.lat]);
      if (!px || !py) return;

      const g = nodeG.append('g')
        .attr('transform', `translate(${px},${py})`)
        .style('cursor', 'crosshair');

      // Pulse ring 1
      const pulse1 = g.append('circle').attr('class', 'node-pulse').attr('r', 4).attr('opacity', 0.5);
      pulse1.append('animate')
        .attr('attributeName', 'r').attr('values', '4;16;4')
        .attr('dur', '2.5s').attr('repeatCount', 'indefinite');

      // Pulse ring 2
      const pulse2 = g.append('circle').attr('class', 'node-pulse').attr('r', 6).attr('opacity', 0.3);
      pulse2.append('animate')
        .attr('attributeName', 'opacity').attr('values', '0.3;0;0.3')
        .attr('dur', '2.5s').attr('repeatCount', 'indefinite');

      // Core dot
      g.append('circle').attr('class', 'node-dot').attr('r', 3);

      // Label
      g.append('text').attr('class', 'node-label').attr('x', 6).attr('y', -5).text(n.label);

      // Tooltip
      g.on('mouseenter', () => {
        tip.classList.remove('hidden');
        tip.style.left = (px + 14) + 'px';
        tip.style.top  = (py - 34) + 'px';
        tip.textContent = `[${n.id}] ${n.label} — ACTIVE NODE`;
      }).on('mouseleave', () => tip.classList.add('hidden'));
    });

    // Container for animated arc dots
    svg.append('g').attr('id', 'arc-dots-g');
  }

  // Animated dots travelling along arcs
  function animateArcs() {
    const dotsG = svg.select('#arc-dots-g');
    if (dotsG.empty() || !worldData) { requestAnimationFrame(animateArcs); return; }

    dotsG.selectAll('*').remove();
    arcT = (arcT + 0.0025) % 1;

    ARC_PAIRS.forEach(([fi, ti], idx) => {
      const n1 = MAP_NODES[fi], n2 = MAP_NODES[ti];
      const t  = (arcT + idx * (1 / ARC_PAIRS.length)) % 1;
      const interp = d3.geoInterpolate([n1.lon, n1.lat], [n2.lon, n2.lat]);
      const [plon, plat] = interp(t);
      const [px, py]     = projection([plon, plat]);
      if (px && py) {
        dotsG.append('circle')
          .attr('class', 'arc-dot')
          .attr('cx', px).attr('cy', py)
          .attr('r', 2.5);
      }
    });
    requestAnimationFrame(animateArcs);
  }

  // Load Natural Earth 110m TopoJSON
  fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
    .then(r => r.json())
    .then(data => {
      worldData = data;
      resize();
      animateArcs();
    })
    .catch(() => {
      svg.append('rect').attr('width', W || 600).attr('height', H || 280).attr('fill', '#000e1c');
      svg.append('text')
        .attr('x', (W||600)/2).attr('y', (H||280)/2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#007a88')
        .attr('font-family', 'Share Tech Mono').attr('font-size', 10)
        .text('MAP DATA UNAVAILABLE — CHECK NETWORK');
    });

  window.addEventListener('resize', resize);
  resize();
}

/* ═══════════════════════════════════════════════════════
   SIGNAL WAVE
═══════════════════════════════════════════════════════ */
function initWave() {
  const canvas = document.getElementById('wave-canvas');
  canvas.width  = canvas.parentElement.clientWidth || 400;
  canvas.height = 52;
  const ctx = canvas.getContext('2d');
  let x = 0;
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = '#00f3ff'; ctx.lineWidth = 1.5;
    ctx.shadowBlur  = 6; ctx.shadowColor = '#00f3ff';
    for (let i = 0; i < canvas.width; i++) {
      const y = 26
        + Math.sin(i * .04 + x)       * 10
        + Math.sin(i * .09 + x * 1.3) * 5
        + Math.sin(i * .02 + x * .7)  * 8;
      i === 0 ? ctx.moveTo(i, y) : ctx.lineTo(i, y);
    }
    ctx.stroke(); ctx.shadowBlur = 0;
    x += .05;
    requestAnimationFrame(draw);
  }
  draw();
}

/* ═══════════════════════════════════════════════════════
   CIRCULAR GAUGES
═══════════════════════════════════════════════════════ */
const CIRC = 239;

function setGauge(id, valId, val) {
  document.getElementById(id).style.strokeDashoffset = CIRC - (CIRC * val / 100);
  document.getElementById(valId).textContent = val + '%';
}

function initGauges() {
  setGauge('g-cpu', 'gv-cpu', 78);
  setGauge('g-mem', 'gv-mem', 61);
  setGauge('g-net', 'gv-net', 45);
}

/* ── Live stat updates ── */
function updateLiveStats() {
  setGauge('g-cpu', 'gv-cpu', Math.floor(Math.random() * 30) + 60);
  setGauge('g-mem', 'gv-mem', Math.floor(Math.random() * 25) + 50);
  setGauge('g-net', 'gv-net', Math.floor(Math.random() * 40) + 30);

  document.getElementById('ms-nodes').textContent = Math.floor(Math.random() * 50) + 220;
  document.getElementById('ms-data').textContent  = (Math.random() * 3 + 3).toFixed(1) + ' TB';

  const tl = document.getElementById('threat-level');
  const r  = Math.random();
  if (r < .6)      { tl.textContent = 'THREAT: LOW';    tl.className = 'threat-low'; }
  else if (r < .85) { tl.textContent = 'THREAT: MEDIUM'; tl.className = 'threat-med'; }
  else              { tl.textContent = 'THREAT: HIGH';   tl.className = 'threat-high'; }

  document.getElementById('r-targets').textContent = String(Math.floor(Math.random()*8)+4).padStart(2,'0');
  document.getElementById('r-locked').textContent  = String(Math.floor(Math.random()*3)+1).padStart(2,'0');
}

/* ═══════════════════════════════════════════════════════
   PHOENIX TERMINAL
═══════════════════════════════════════════════════════ */
function initPhxTerminal() {
  const output = document.getElementById('term-output');
  const input  = document.getElementById('term-input');
  let history = [], histIdx = -1;

  function println(text, cls = 'out-line') {
    text.split('\n').forEach(l => {
      const p = document.createElement('p');
      p.className   = 'term-line ' + cls;
      p.textContent = l;
      output.appendChild(p);
    });
    output.scrollTop = output.scrollHeight;
  }

  println('PHOENIX OS v9.1 — SECURE TERMINAL');
  println('Type HELP for available commands.');
  println('');

  const COMMANDS = {
    help() {
      println('╔═══════════════════════════════════════╗');
      println('║         PHOENIX TERMINAL HELP         ║');
      println('╠═══════════════════════════════════════╣');
      println('║ HELP       — Show this message        ║');
      println('║ STATUS     — Node status report       ║');
      println('║ SCAN       — Network scan             ║');
      println('║ PING <ip>  — Ping a target            ║');
      println('║ WHOAMI     — Current operator info    ║');
      println('║ NODES      — List active nodes        ║');
      println('║ DECRYPT    — Decrypt last intercept   ║');
      println('║ TRACE <ip> — Trace IP origin          ║');
      println('║ UPLINK     — Check satellite uplink   ║');
      println('║ CLEAR      — Clear terminal           ║');
      println('║ DATE       — Show current timestamp   ║');
      println('║ SYSINFO    — System information       ║');
      println('║ MATRIX     — Matrix rain effect       ║');
      println('╚═══════════════════════════════════════╝');
    },
    status() {
      println(`NODE      : PHX-ALPHA`);
      println(`UPTIME    : ${Math.floor(Math.random()*999+100)}h ${Math.floor(Math.random()*59)}m`);
      println(`CPU LOAD  : ${Math.floor(Math.random()*30)+60}%`);
      println(`MEM USAGE : ${Math.floor(Math.random()*25)+50}%`);
      println(`THREAT LVL: ${['LOW','LOW','MEDIUM','HIGH'][Math.floor(Math.random()*4)]}`);
      println(`ENCRYPTION: AES-512 ACTIVE`,   'ok-line');
      println(`STATUS    : NOMINAL`,           'ok-line');
    },
    scan() {
      println('Initiating network scan...');
      setTimeout(() => {
        const found = Math.floor(Math.random() * 8) + 5;
        println(`Scan complete. ${found} hosts detected:`);
        for (let i = 0; i < found; i++) {
          const ip    = randIP();
          const ports = ['22/SSH','443/HTTPS','80/HTTP','3306/MySQL','8080/HTTP-ALT'][Math.floor(Math.random()*5)];
          println(`  ${ip.padEnd(18)} PORT ${ports}`);
        }
        println('Scan finished.', 'ok-line');
        output.scrollTop = output.scrollHeight;
      }, 800);
    },
    ping(args) {
      const target = args || randIP();
      println(`Pinging ${target}...`);
      let count = 0;
      const iv = setInterval(() => {
        const ms = Math.floor(Math.random() * 120) + 8;
        println(`Reply from ${target}: bytes=64 time=${ms}ms TTL=64`, 'ok-line');
        if (++count >= 4) {
          clearInterval(iv);
          println(`Ping stats: 4 sent, 4 received, 0% loss.`);
        }
        output.scrollTop = output.scrollHeight;
      }, 300);
    },
    whoami() {
      println('Operator  : PHOENIX');
      println('Clearance : LEVEL 9 — UNRESTRICTED');
      println('Node      : PHX-ALPHA');
      println('Session   : ENCRYPTED / ANONYMOUS');
      println('Location  : [CLASSIFIED]');
    },
    nodes() {
      println(`Active surveillance nodes (${Math.floor(Math.random()*50)+220} online):`);
      MAP_NODES.forEach(n => {
        const load = Math.floor(Math.random() * 60) + 20;
        println(`  [${n.id}] ${n.label.padEnd(12)} LON:${n.lon.toFixed(1)} LAT:${n.lat.toFixed(1)} LOAD:${load}%`);
      });
    },
    decrypt() {
      println('Loading cipher buffer...');
      setTimeout(() => {
        const msgs = [
          '"Operation confirmed for 0400. Move on signal."',
          '"Transfer 8.4M — no trace."',
          '"The asset is in play."',
          '"Rendezvous at grid 447-B."',
        ];
        println(`KEY: ${Math.random().toString(16).substr(2,16).toUpperCase()}`, 'ok-line');
        println(`MSG: ${msgs[Math.floor(Math.random()*msgs.length)]}`,           'ok-line');
        output.scrollTop = output.scrollHeight;
      }, 600);
    },
    trace(args) {
      const target = args || randIP();
      println(`Tracing ${target}...`);
      let delay = 0;
      const hops = Math.floor(Math.random() * 6) + 5;
      for (let i = 1; i <= hops; i++) {
        const ms  = Math.floor(Math.random() * 80) + 5;
        const hop = randIP();
        setTimeout(() => {
          println(`  ${String(i).padStart(2)}  ${hop.padEnd(18)} ${ms}ms`);
          output.scrollTop = output.scrollHeight;
        }, delay);
        delay += 250;
      }
      setTimeout(() => {
        const isp = ['Verizon','BT Group','Rostelecom','China Telecom'][Math.floor(Math.random()*4)];
        println(`Target resolved: ISP=${isp} ORIGIN=MASKED`, 'ok-line');
        output.scrollTop = output.scrollHeight;
      }, delay + 100);
    },
    uplink() {
      println('Checking satellite uplink...');
      setTimeout(() => {
        const sats = ['PHOENIX-7','ARGUS-3','HELIOS-12','SHADOW-NET-4'];
        const sat  = sats[Math.floor(Math.random() * sats.length)];
        println(`SATELLITE : ${sat}`,                                       'ok-line');
        println(`ALT       : ${(Math.random()*200+400).toFixed(0)} km`,    'ok-line');
        println(`BANDWIDTH : ${(Math.random()*500+100).toFixed(0)} Mbps`, 'ok-line');
        println(`LATENCY   : ${(Math.random()*20+5).toFixed(1)} ms`,       'ok-line');
        println(`STATUS    : ENCRYPTED UPLINK LIVE`,                        'ok-line');
        output.scrollTop = output.scrollHeight;
      }, 700);
    },
    clear() { output.innerHTML = ''; },
    date()  { println(new Date().toISOString() + ' UTC'); },
    sysinfo() {
      println('PHOENIX OS v9.1.3 — CLASSIFIED BUILD');
      println(`KERNEL  : PHX-CORE 4.9.1`);
      println(`ARCH    : x86_64-classified`);
      println(`NODES   : ${Math.floor(Math.random()*50)+220} ACTIVE`);
      println(`DATA    : ${(Math.random()*3+3).toFixed(1)} TB intercepted today`);
      println(`UPTIME  : ${Math.floor(Math.random()*99)+1}d ${Math.floor(Math.random()*23)}h`);
    },
    matrix() {
      println('MATRIX RAIN INITIATED — auto-stops after 5s');
      let running = true;
      const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
      const iv = setInterval(() => {
        let line = '';
        for (let i = 0; i < 40; i++) line += chars[Math.floor(Math.random() * chars.length)];
        println(line, 'ok-line');
        output.scrollTop = output.scrollHeight;
        if (!running) clearInterval(iv);
      }, 80);
      setTimeout(() => { running = false; clearInterval(iv); println('[MATRIX RAIN STOPPED]'); }, 5000);
    },
  };

  function handleCommand(raw) {
    const trimmed = raw.trim();
    if (!trimmed) return;
    history.unshift(trimmed); histIdx = -1;
    println(`PHX@ALPHA:~$ ${trimmed}`, 'input-line');
    const parts = trimmed.toUpperCase().split(/\s+/);
    const cmd   = parts[0];
    const args  = parts.slice(1).join(' ');
    if (COMMANDS[cmd.toLowerCase()]) {
      try { COMMANDS[cmd.toLowerCase()](args || null); }
      catch (e) { println('Error: ' + e.message, 'err-line'); }
    } else {
      println(`Command not found: ${cmd}. Type HELP for commands.`, 'err-line');
    }
    println('');
  }

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const val = input.value; input.value = ''; handleCommand(val);
    } else if (e.key === 'ArrowUp') {
      if (histIdx < history.length - 1) { histIdx++; input.value = history[histIdx] || ''; }
    } else if (e.key === 'ArrowDown') {
      if (histIdx > 0) { histIdx--; input.value = history[histIdx] || ''; }
      else { histIdx = -1; input.value = ''; }
    } else if (e.key === 'Tab') {
      e.preventDefault();
      const partial  = input.value.trim().toLowerCase();
      const matches  = Object.keys(COMMANDS).filter(c => c.startsWith(partial));
      if (matches.length === 1)     { input.value = matches[0].toUpperCase() + ' '; }
      else if (matches.length > 1)  { println(matches.join('  '), 'warn-line'); }
    } else if (e.key === 'c' && e.ctrlKey) {
      input.value = ''; println('^C', 'err-line');
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault(); output.innerHTML = '';
    }
  });

  document.addEventListener('click', () => input.focus());
  input.focus();
}

/* ═══════════════════════════════════════════════════════
   OPERATION BUTTONS
═══════════════════════════════════════════════════════ */
function randIP()  { return Array.from({length:4}, () => Math.floor(Math.random()*254)+1).join('.'); }
function randMac() { return Array.from({length:6}, () => Math.floor(Math.random()*256).toString(16).padStart(2,'0').toUpperCase()).join(':'); }

const OPS = {
  coords: {
    title: 'TRIANGULATING COORDINATES',
    steps: ['Pinging satellite array...','Resolving GPS vectors...','Cross-referencing cell towers...','Triangulation complete.'],
    result() {
      const lat = (Math.random()*180-90).toFixed(4), lng = (Math.random()*360-180).toFixed(4);
      const cities = ['MOSCOW','TOKYO','CHICAGO','DUBAI','CAIRO','LIMA','OSLO'];
      const city   = cities[Math.floor(Math.random()*cities.length)];
      return `\n◉ TARGET COORDINATES LOCKED\n  LAT: ${lat}°\n  LNG: ${lng}°\n  NEAREST CITY: ${city}\n  ACCURACY: ${(Math.random()*5+95).toFixed(1)}%\n  TIMESTAMP: ${new Date().toISOString()}`;
    }
  },
  'hack-phones': {
    title: 'INITIATING MASS DEVICE BREACH',
    steps: ['Broadcasting exploit payload...','Scanning 4G/5G frequencies...','Injecting zero-day CVE-2024-8821...','Breaking WPA3 handshakes...','Compromising IMEI database...','Extracting contact manifests...','Mirroring SMS threads...','Breach complete.'],
    result() {
      const names   = ['James Kowalski','Priya Mehta','Dmitri Volkov','Amara Diallo','Chen Wei','Sofia Reyes'];
      const phones  = ['+1 312 555 0182','+44 7700 900462','+7 495 555 0031','+254 700 123456','+86 139 0000 1234','+52 55 5555 0187'];
      const devices = ['iPhone 16 Pro','Samsung Galaxy S25','Pixel 9','OnePlus 13','Xiaomi 14'];
      let out = '\n◉ BREACH COMPLETE — COMPROMISED DEVICES:\n\n';
      for (let i = 0; i < 5; i++) {
        const n    = names[Math.floor(Math.random()*names.length)];
        const p    = phones[Math.floor(Math.random()*phones.length)];
        const d    = devices[Math.floor(Math.random()*devices.length)];
        const msgs = ['Tell no one','Meeting at 0300','Package delivered','The key is 7741','Do not reply'];
        out += `  [${String(i+1).padStart(2,'0')}] ${n}\n       ${p} | ${d}\n       LAST MSG: "${msgs[Math.floor(Math.random()*msgs.length)]}"\n\n`;
      }
      return out;
    }
  },
  'sat-link': {
    title: 'ESTABLISHING SATELLITE LINK',
    steps: ['Targeting LEO cluster PHX-7...','Rotating dish bearing 042°...','Handshake protocol v4.1...','Encryption tunnel established.'],
    result() {
      const sat = ['PHOENIX-7','ARGUS-3','HELIOS-12','SHADOW-NET-4'];
      return `\n◉ SAT LINK ACTIVE\n  SATELLITE: ${sat[Math.floor(Math.random()*sat.length)]}\n  ORBIT ALT: ${(Math.random()*200+400).toFixed(0)} km\n  BANDWIDTH: ${(Math.random()*500+100).toFixed(0)} Mbps\n  LATENCY: ${(Math.random()*20+5).toFixed(1)} ms\n  STATUS: ENCRYPTED UPLINK LIVE`;
    }
  },
  blackout: {
    title: 'POWER GRID DISRUPTION SEQUENCE',
    steps: ['Locating grid control nodes...','Spoofing SCADA credentials...','Injecting rogue commands...','Rolling blackout initiated.'],
    result() {
      const zones = ['Grid Zone 7-Alpha','District 9 Eastern Hub','Industrial Sector 3','Coastal Relay Station B'];
      return `\n◉ BLACKOUT TRIGGERED\n  ZONE: ${zones[Math.floor(Math.random()*zones.length)]}\n  AFFECTED: ${Math.floor(Math.random()*900+100)}k users\n  DURATION EST: ${Math.floor(Math.random()*60+5)} minutes\n  TRACEBACK: NULLIFIED\n  ORIGIN MASKED: ✓`;
    }
  },
  decrypt: {
    title: 'DECRYPTING INTERCEPTED UPLINK',
    steps: ['Loading cipher text...','Running brute-force AES...','Trying RSA-4096 keyset...','Applying rainbow table...','Decryption successful.'],
    result() {
      const msgs = ['"Operation confirmed for 0400. Move on signal."','"Transfer 8.4M to account ████████. No trace."','"The asset is in play. Await confirmation."','"Rendezvous at grid 447-B. Come alone."'];
      return `\n◉ UPLINK DECRYPTED\n  CIPHER: AES-256-CBC (broken)\n  KEY HASH: ${Math.random().toString(16).substr(2,16).toUpperCase()}\n  PLAINTEXT:\n  ${msgs[Math.floor(Math.random()*msgs.length)]}\n  SENDER IP: ${randIP()}\n  TIMESTAMP: ${new Date().toISOString()}`;
    }
  },
  wipe: {
    title: 'REMOTE WIPE SEQUENCE',
    steps: ['Acquiring target device UID...','Sending kill command...','Overwriting storage sectors...','Device memory zeroed.'],
    result() {
      const targets = ['DEVICE-4F2A','LAPTOP-91B3','TABLET-7C0D','SERVER-221E'];
      return `\n◉ REMOTE WIPE COMPLETE\n  TARGET: ${targets[Math.floor(Math.random()*targets.length)]}\n  DATA DESTROYED: ${(Math.random()*500+50).toFixed(1)} GB\n  PASSES: 7x DOD 5220.22\n  RECOVERY POSSIBLE: NO\n  DENIABILITY: FULL`;
    }
  },
  trace: {
    title: 'TRACING IP SOURCE',
    steps: ['Initiating traceroute...','Peeling proxy layers...','Bypassing Tor exit nodes...','Physical location resolved.'],
    result() {
      const isp = ['Verizon','BT Group','Rostelecom','China Telecom','Airtel'];
      return `\n◉ IP TRACED\n  ADDRESS: ${randIP()}\n  ISP: ${isp[Math.floor(Math.random()*isp.length)]}\n  LOCATION: ${(Math.random()*180-90).toFixed(2)}°, ${(Math.random()*360-180).toFixed(2)}°\n  MAC: ${randMac()}\n  DEVICE: ${['Linux x86_64','Windows 11','macOS Ventura'][Math.floor(Math.random()*3)]}\n  PROXY LAYERS PEELED: ${Math.floor(Math.random()*8+3)}`;
    }
  },
  intercept: {
    title: 'INTERCEPTING COMMUNICATIONS',
    steps: ['Tapping fiber relay node...','Spoofing TLS certificate...','MITM tunnel active...','Streaming comms buffer...'],
    result() {
      const convos = ['"Did you transfer it?" / "Yes, done. Delete this."','"They suspect nothing. Proceed as planned."','"Location confirmed. ETA 0220 hours."','"Package is hot. Do NOT open before 0600."'];
      return `\n◉ COMMS INTERCEPTED\n  PROTOCOL: TLS 1.3 (spoofed)\n  ENDPOINT A: ${randIP()}\n  ENDPOINT B: ${randIP()}\n  SNIPPET:\n  ${convos[Math.floor(Math.random()*convos.length)]}\n  FULL LOG: 2.${Math.floor(Math.random()*9+1)} MB captured`;
    }
  },
};

function initOpsButtons() {
  document.querySelectorAll('.op-btn').forEach(btn => {
    btn.addEventListener('click', () => runOp(btn.dataset.op));
  });
}

function runOp(opId) {
  const op = OPS[opId]; if (!op) return;
  document.querySelectorAll('.op-btn').forEach(b => b.classList.add('busy'));

  const modal = document.getElementById('op-modal');
  const title = document.getElementById('op-modal-title');
  const fill  = document.getElementById('op-modal-fill');
  const log   = document.getElementById('op-modal-log');

  modal.classList.remove('hidden');
  title.textContent = op.title;
  fill.style.width  = '0%';
  log.innerHTML     = '';

  let step = 0;
  const total = op.steps.length;

  function nextStep() {
    if (step >= total) {
      fill.style.width = '100%';
      setTimeout(() => {
        modal.classList.add('hidden');
        document.querySelectorAll('.op-btn').forEach(b => b.classList.remove('busy'));
        const out = document.getElementById('op-output');
        out.innerHTML   = '';
        const result    = op.result();
        let ci = 0;
        function typeChar() {
          if (ci < result.length) {
            out.textContent += result[ci++];
            out.scrollTop    = out.scrollHeight;
            setTimeout(typeChar, 18);
          }
        }
        typeChar();
      }, 600);
      return;
    }
    fill.style.width = Math.round((step / total) * 100) + '%';
    const d = document.createElement('div');
    d.textContent = '> ' + op.steps[step];
    log.appendChild(d); log.scrollTop = log.scrollHeight;
    step++;
    setTimeout(nextStep, 400 + Math.random() * 400);
  }
  nextStep();
}

/* ═══════════════════════════════════════════════════════
   BOOTSTRAP
═══════════════════════════════════════════════════════ */
window.addEventListener('resize', resizeBg);
resizeBg();
for (let i = 0; i < 100; i++) particles.push(new Particle());
animateBg();
drawTesseract();
runLoading();