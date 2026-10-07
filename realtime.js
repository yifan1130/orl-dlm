'use strict';

// Real-time replay of recorded wall-clock decoding: each lane answers the same problems back to back and text appears
// when it was emitted. Sequential lanes list per streamed chunk its token count and arrival time; the block-diffusion
// lane lists per block its emission time, step count and every position's settle round. The card has two faces, one
// recording per model family, and flips between them.
(() => {
  const root = document.querySelector('.rt-demo');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const FILL = 'abcdefghijklmnopqrstuvwxyz0123456789#$%&*+=<>?/|~^';
  const COLOR = { ar: '#a3abb7', opd: '#5d6878', orl: '#2475db' };
  const FRESH_S = 0.12, HOLD_S = 2.5, FRAME_MS = 33, STEP_S = 0.05, FLIP_MS = 220;
  const FACES = { qwen: 'assets/orl-realtime.json?v=20261005e', gemma: 'assets/orl-realtime-gemma.json?v=20261007b' };
  const chart = $('rt-chart'), play = $('rt-play');
  const head = { badge: root.querySelector('.rt-head .demo-badge'), title: $('rt-title'), lead: root.querySelector('.rt-head p') };
  const defaults = Object.fromEntries(Object.entries(head).map(([k, node]) => [k, node.textContent]));
  const loaded = {};
  let lanes = [], windowS = 20, t = 0, timer = null, last = 0, series = [], most = 1, face = null, flipping = false;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  const compact = n => (n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n));

  // Number of entries <= time in an ascending array.
  function upto(sorted, time) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= time) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function placeholder(final, seed) {
    let out = '', i = 0;
    for (const c of final) {
      out += /\s/.test(c) ? c : FILL[Math.abs(seed * 7 + i * 13) % FILL.length];
      i++;
    }
    return out;
  }

  function shell(spec) {
    const card = element('article', `rt-lane lane-${spec.key}`);
    const head = element('header');
    head.append(element('h3', '', spec.name), element('span', '', spec.detail));
    const stats = element('div', 'rt-stats');
    const rate = element('b', '', '–');
    stats.append(rate, element('small', '', 'tokens / s'));
    const solved = element('div', 'rt-solved');
    const question = element('div', 'rt-question');
    const pane = element('pre', 'rt-text');
    pane.tabIndex = 0;
    pane.setAttribute('aria-label', `${spec.name} output`);
    card.append(head, stats, solved, question, pane);
    $('rt-lanes').append(card);
    return { rate, solved, question, pane, current: -1, block: -1, spans: [], done: null, live: null };
  }

  function build(spec) {
    let t0 = 0;
    const reveals = [];
    const problems = spec.problems.map(p => {
      const q = { question: p.question, correct: p.correct, answer: p.answer, t0, t1: t0 + p.wall };
      if (p.blocks) {
        let start = t0 + (p.prefill || 0);
        q.blocks = p.blocks.map(b => {
          const end = t0 + b.end, dt = (end - start) / (b.steps + 1), hidden = new Set(b.hidden);
          const text = b.tokens.map((i, j) => (hidden.has(j) ? '' : spec.vocab[i]));
          const reveal = b.settle.map(s => start + Math.max(s, 1) * dt);
          text.forEach((s, j) => { if (s) reveals.push(reveal[j]); });
          const block = { start, end, dt, text, reveal, joined: text.join('') };
          start = end;
          return block;
        });
      } else {
        q.text = p.tokens.map(i => spec.vocab[i]);
        q.reveal = [];
        let n = 0;
        p.chunks.forEach((size, c) => {
          for (let j = 0; j < size; j++) q.reveal[n++] = t0 + p.times[c];
        });
        q.text.forEach((s, j) => { if (s) reveals.push(q.reveal[j]); });
      }
      t0 += p.wall;
      return q;
    });
    reveals.sort((a, b) => a - b);
    return { ...spec, problems, reveals, ui: shell(spec) };
  }

  function start(lane, index) {
    const ui = lane.ui, p = lane.problems[index];
    ui.current = index;
    ui.block = -1;
    ui.question.textContent = `Q${index + 1}  ${p.question}`;
    ui.done = document.createTextNode('');
    ui.live = element('span', p.blocks ? '' : 'fresh');
    ui.pane.replaceChildren(ui.done, ui.live);
  }

  function renderSequential(p, ui, time) {
    const n = upto(p.reveal, time), fresh = upto(p.reveal, time - FRESH_S);
    ui.done.data = p.text.slice(0, fresh).join('');
    ui.live.textContent = p.text.slice(fresh, n).join('');
  }

  function renderCanvas(p, ui, time) {
    let active = 0;
    while (active + 1 < p.blocks.length && p.blocks[active + 1].start <= time) active++;
    const b = p.blocks[active];
    if (active !== ui.block) {
      ui.block = active;
      ui.done.data = p.blocks.slice(0, active).map(x => x.joined).join('');
      ui.spans = b.text.map(() => element('span'));
      ui.live.replaceChildren(...ui.spans);
    }
    const step = Math.max(0, Math.floor((time - b.start) / b.dt));
    b.text.forEach((text, j) => {
      if (!text) return;
      const span = ui.spans[j];
      if (b.reveal[j] <= time) {
        span.textContent = text;
        span.className = b.reveal[j] > time - FRESH_S && text.trim() ? 'fresh' : '';
      } else {
        span.textContent = placeholder(text, j + step * 131);
        span.className = 'noise';
      }
    });
  }

  function renderLane(lane, time) {
    const ui = lane.ui, problems = lane.problems;
    const shown = upto(lane.reveals, time);
    const busy = Math.min(time, problems[problems.length - 1].t1);  // the rate stops when the lane runs out of problems
    ui.rate.textContent = busy > 0.4 ? Math.round(shown / busy) : '–';
    let index = problems.findIndex(p => p.t1 > time);
    const finishedAll = index < 0;
    if (finishedAll) index = problems.length - 1;
    const finished = problems.filter(p => p.t1 <= time), right = finished.filter(p => p.correct).length;
    ui.solved.innerHTML = `<span><b>${finished.length}</b> solved</span><span class="ok">✓ ${right}</span>` +
      (finished.length > right ? `<span class="bad">✗ ${finished.length - right}</span>` : '') +
      (finishedAll ? `<span class="all">all ${problems.length} done at ${problems[problems.length - 1].t1.toFixed(1)} s</span>` : '');
    if (index !== ui.current) start(lane, index);
    const p = problems[index];
    if (p.blocks) renderCanvas(p, ui, time);
    else renderSequential(p, ui, time);
    ui.pane.scrollTop = ui.pane.scrollHeight;
    return shown;
  }

  function layoutChart() {
    const width = Math.max(320, Math.round(chart.clientWidth || 900));
    chart.setAttribute('viewBox', `0 0 ${width} 132`);
    chart.replaceChildren();
    const ns = 'http://www.w3.org/2000/svg', box = { left: 44, right: width - 64, top: 12, bottom: 108 };
    const make = (tag, attrs, text) => {
      const node = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
      if (text != null) node.textContent = text;
      chart.append(node);
      return node;
    };
    make('line', { x1: box.left, x2: box.right, y1: box.bottom, y2: box.bottom, class: 'axis' });
    for (let s = 0; s <= windowS; s += 5) {
      const x = box.left + (box.right - box.left) * s / windowS;
      make('text', { x, y: 126, class: 'tick', 'text-anchor': 'middle' }, `${s} s`);
    }
    make('text', { x: box.left - 8, y: box.top + 8, class: 'tick', 'text-anchor': 'end' }, compact(most));
    make('text', { x: box.left - 8, y: box.bottom, class: 'tick', 'text-anchor': 'end' }, '0');
    make('text', { x: box.left, y: box.top - 2, class: 'tick' }, 'tokens generated');
    chart.box = box;
    chart.paths = lanes.map(lane => make('path', { class: `line line-${lane.key}`, stroke: COLOR[lane.key] }));
    chart.labels = lanes.map(lane => make('text', { class: `end end-${lane.key}`, fill: COLOR[lane.key] }));
  }

  function renderChart(time) {
    const { box } = chart;
    const x = s => box.left + (box.right - box.left) * s / windowS;
    const y = n => box.bottom - (box.bottom - box.top) * n / most;
    const ends = lanes.map((lane, i) => {
      const points = [], samples = series[i], end = Math.floor(time / STEP_S);
      for (let k = 0; k <= end && k < samples.length; k++) points.push(`${x(k * STEP_S).toFixed(1)},${y(samples[k]).toFixed(1)}`);
      const now = upto(lane.reveals, time);
      points.push(`${x(time).toFixed(1)},${y(now).toFixed(1)}`);
      chart.paths[i].setAttribute('d', `M${points.join('L')}`);
      return { i, now, y: y(now) + 4 };
    });
    ends.sort((a, b) => a.y - b.y);  // keep the end labels at least 13 px apart, top to bottom
    ends.forEach((e, k) => { if (k) e.y = Math.max(e.y, ends[k - 1].y + 13); });
    ends.forEach(({ i, now, y: labelY }) => {
      const label = chart.labels[i];
      label.setAttribute('x', (x(time) + 6).toFixed(1));
      label.setAttribute('y', labelY.toFixed(1));
      label.textContent = compact(now);
    });
  }

  function render(time) {
    const clipped = Math.min(time, windowS);
    lanes.forEach(lane => renderLane(lane, clipped));
    renderChart(clipped);
    $('rt-clock').textContent = `${clipped.toFixed(1)} s`;
  }

  function stop() {
    clearTimeout(timer);
    timer = null;
    root.classList.remove('playing');
    play.setAttribute('aria-label', 'Play');
  }

  function tick() {
    const now = performance.now();
    t += (now - last) / 1000;
    last = now;
    if (t > windowS + HOLD_S) t = 0;
    render(t);
    timer = setTimeout(tick, FRAME_MS);
  }

  function begin() {
    if (t >= windowS) t = 0;
    last = performance.now();
    root.classList.add('playing');
    play.setAttribute('aria-label', 'Pause');
    timer = setTimeout(tick, FRAME_MS);
  }

  function load(key) {
    loaded[key] = loaded[key] || fetch(FACES[key]).then(response => {
      if (!response.ok) throw new Error(response.statusText);
      return response.json();
    });
    return loaded[key];
  }

  // Rebuild the lanes, chart, note and heading from one recording.
  function show(data) {
    windowS = data.window;
    $('rt-lanes').replaceChildren();
    lanes = data.lanes.map(build);
    series = lanes.map(lane => {
      const samples = [];
      for (let s = 0; s <= windowS + 1e-9; s += STEP_S) samples.push(upto(lane.reveals, s));
      return samples;
    });
    most = Math.max(1, ...series.map(s => s[s.length - 1]));
    $('rt-note').textContent = data.note;
    const text = data.head || defaults;
    Object.entries(head).forEach(([k, node]) => { node.textContent = text[k]; });
    layoutChart();
    t = reduceMotion ? windowS : 0;
    render(t);
  }

  const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

  async function flip(key) {
    if (key === face || flipping) return;
    flipping = true;
    root.querySelectorAll('.rt-flip button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.face === key)));
    const data = load(key);
    stop();
    if (!reduceMotion) {
      root.classList.add('flip-out');
      await wait(FLIP_MS);
    }
    try {
      show(await data);
      face = key;
    } catch {
      $('rt-note').textContent = 'The recorded runs could not be loaded.';
    }
    if (!reduceMotion) {
      root.classList.replace('flip-out', 'flip-in');
      void root.offsetWidth;  // start the turn back from the far side
      root.classList.remove('flip-in');
      begin();
    }
    flipping = false;
  }

  play.addEventListener('click', () => (timer ? stop() : begin()));
  window.addEventListener('resize', () => { if (lanes.length) { layoutChart(); render(t); } });
  root.querySelectorAll('.rt-flip button').forEach(b => b.addEventListener('click', () => flip(b.dataset.face)));
  load('qwen')
    .then(data => {
      show(data);
      face = 'qwen';
      if (reduceMotion || !('IntersectionObserver' in window)) return;
      const observer = new IntersectionObserver(entries => {
        if (entries.some(entry => entry.isIntersecting)) {
          observer.disconnect();
          if (!timer) begin();
        }
      }, { threshold: 0.3 });
      observer.observe(root);
    })
    .catch(() => {
      $('rt-note').textContent = 'The recorded runs could not be loaded.';
    });
})();
