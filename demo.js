'use strict';

// Forward-by-forward comparison of three recorded generations of one problem (assets/orl-race.json). Sequential
// lanes list their tokens and how many each forward pass emitted; the block-diffusion lane lists, per block, the
// final token ids and the canvas each denoising step started from (a block takes its steps plus one cache commit).
(() => {
  const root = document.querySelector('.race-demo');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const slider = $('race-slider'), play = $('race-play');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DURATION_MS = 16000;
  const WIDE = /[\u1100-\u115F\u2E80-\u303E\u3041-\u33FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7A3\uF900-\uFAFF\uFF00-\uFF60\uFFE0-\uFFE6]/;
  const NARROW = /[\u0021-\u007E\u00A1-\u024F\u0370-\u03FF\u0400-\u04FF]/;
  const FILL = 'abcdefghijklmnopqrstuvwxyz0123456789#$%&*+=<>?/|~^';

  let lanes = [], most = 1, timer = null;

  // The final text's layout, drawn with the held token's own characters where they fit (filler for special tokens).
  function noise(final, token, seed) {
    const special = /^<\|.*\|>$/.test(token);
    const narrow = special ? [] : [...token].filter(c => NARROW.test(c));
    const wide = special ? [] : [...token].filter(c => WIDE.test(c));
    let out = '', i = 0;
    for (const c of final) {
      if (/\s/.test(c)) out += c;
      else if (WIDE.test(c)) out += wide.length ? wide[i % wide.length] : String.fromCharCode(0x4E00 + (seed * 31 + i * 97) % 6000);
      else out += narrow.length ? narrow[(seed + i) % narrow.length] : FILL[(seed * 7 + i * 13) % FILL.length];
      i++;
    }
    return out;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function laneShell(spec) {
    const card = element('article', `race-lane lane-${spec.key}`);
    const head = element('header');
    head.append(element('h3', '', spec.name), element('span', '', spec.detail));
    const stats = element('div', 'lane-stats');
    const count = element('b', '', '0');
    stats.append(count, element('small', '', `of ${spec.shown} tokens · ${(spec.shown / spec.forwards).toFixed(1)} per forward`));
    const bar = element('div', 'lane-bar');
    const fill = element('i');
    bar.append(fill);
    const text = element('pre', 'lane-text');
    text.tabIndex = 0;
    text.setAttribute('aria-label', `${spec.name} output at the selected forward pass`);
    const status = element('footer', 'lane-status');
    card.append(head, stats, bar, text, status);
    $('race-lanes').append(card);
    return { count, fill, text, status };
  }

  // Sequential decoding: tokens[i] text ("" for EOS), steps[f] tokens emitted by forward f + 1.
  function sequentialLane(spec) {
    const steps = spec.steps || spec.tokens.map(() => 1);
    const cum = [0], offsets = [0], shownBefore = [0];
    steps.forEach(n => cum.push(cum[cum.length - 1] + n));
    spec.tokens.forEach(t => {
      offsets.push(offsets[offsets.length - 1] + t.length);
      shownBefore.push(shownBefore[shownBefore.length - 1] + (t ? 1 : 0));
    });
    const full = spec.tokens.join('');
    const lane = { ...spec, forwards: steps.length, shown: shownBefore[shownBefore.length - 1] };
    const ui = laneShell(lane);
    const done = document.createTextNode(''), fresh = element('span', 'fresh');
    ui.text.append(done, fresh);
    lane.render = f => {
      const k = Math.min(f, lane.forwards), n = cum[k], before = f <= lane.forwards && k > 0 ? cum[k - 1] : n;
      done.data = full.slice(0, offsets[before]);
      fresh.textContent = full.slice(offsets[before], offsets[n]);
      return finish(lane, ui, f, shownBefore[n]);
    };
    return lane;
  }

  // Block diffusion: per block the final ids, every step's input canvas and the hidden (EOS) positions.
  function canvasLane(spec) {
    let start = 0, shown = 0;
    const blocks = spec.blocks.map(block => {
      const hidden = new Set(block.hidden), rounds = block.frames.length;
      const finals = block.final.map((id, j) => (hidden.has(j) ? '' : spec.vocab[id]));
      const b = { block, rounds, start, finals, text: finals.join(''), shown: finals.filter(Boolean).length };
      start += rounds + 1;
      shown += b.shown;
      return b;
    });
    const lane = { ...spec, forwards: start, shown };
    const ui = laneShell(lane);
    const done = document.createTextNode(''), live = element('span');
    ui.text.append(done, live);
    let active = -1, spans = [];
    lane.render = f => {
      let current = 0;
      while (current + 1 < blocks.length && blocks[current + 1].start <= f) current++;
      const b = blocks[current];
      if (current !== active) {
        active = current;
        done.data = blocks.slice(0, current).map(x => x.text).join('');
        spans = b.finals.map(() => element('span'));
        live.replaceChildren(...spans);
      }
      const k = Math.min(f - b.start, b.rounds), at = (s, j) => (s < b.rounds ? b.block.frames[s][j] : b.block.final[j]);
      let decided = blocks.slice(0, current).reduce((sum, x) => sum + x.shown, 0);
      b.finals.forEach((final, j) => {
        const span = spans[j], id = b.block.final[j];
        if (!final) return;
        if (at(k, j) === id) {
          decided++;
          span.textContent = final;
          span.className = f - b.start <= b.rounds && k > 0 && at(k - 1, j) !== id && final.trim() ? 'fresh' : '';
        } else {
          span.textContent = noise(final, spec.vocab[at(k, j)], j + k * 131 + current * 977);
          span.className = 'noise';
        }
      });
      return finish(lane, ui, f, decided);
    };
    return lane;
  }

  function finish(lane, ui, f, shown) {
    ui.count.textContent = shown;
    ui.fill.style.width = `${100 * shown / lane.shown}%`;
    const complete = f >= lane.forwards, fewer = most / lane.forwards;
    ui.status.textContent = !complete ? 'Generating…'
      : `Done in ${lane.forwards} passes${fewer > 1.05 ? ` · ${fewer.toFixed(1)}× fewer` : ''} · answer ${lane.answer} ✓`;
    ui.status.classList.toggle('complete', complete);
    ui.text.scrollTop = ui.text.scrollHeight;
    return complete;
  }

  const forwardsAt = x => Math.round(most * (x / 1000) ** 2);
  const positionOf = f => 1000 * Math.sqrt(f / most);

  function render(x) {
    const f = forwardsAt(x);
    lanes.forEach(lane => lane.render(f));
    slider.value = x;
    slider.setAttribute('aria-valuetext', `${f} forward passes`);
    $('race-forward').textContent = f;
    $('race-fill').style.width = `${x / 10}%`;
    $('race-thumb').style.left = `${x / 10}%`;
  }

  function stop() {
    clearTimeout(timer);
    timer = null;
    root.classList.remove('playing');
    play.setAttribute('aria-label', 'Play');
  }

  function start() {
    let x = Number(slider.value);
    if (x >= 1000) x = 0;
    let last = performance.now();
    root.classList.add('playing');
    play.setAttribute('aria-label', 'Pause');
    const step = () => {
      const now = performance.now();
      x = Math.min(1000, x + 1000 * (now - last) / DURATION_MS);
      last = now;
      render(x);
      if (x >= 1000) return stop();
      timer = setTimeout(step, 30);
    };
    timer = setTimeout(step, 30);
  }

  function init(data) {
    $('race-question').textContent = data.question;
    lanes = data.lanes.map(spec => (spec.blocks ? canvasLane(spec) : sequentialLane(spec)));
    most = Math.max(...lanes.map(lane => lane.forwards));
    const marks = $('race-marks');
    lanes.forEach(lane => {
      const mark = element('span', `mark-${lane.key}`, `${lane.short} · ${lane.forwards}`), x = positionOf(lane.forwards);
      mark.style.left = `${x / 10}%`;
      mark.classList.toggle('end', x > 900);
      marks.append(mark);
    });
    $('race-note').textContent = data.note;
    render(reduceMotion ? 1000 : 0);
    play.addEventListener('click', () => (timer ? stop() : start()));
    slider.addEventListener('input', () => {
      stop();
      render(Number(slider.value));
    });
    if (reduceMotion || !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        observer.disconnect();
        if (Number(slider.value) === 0) start();
      }
    }, { threshold: 0.35 });
    observer.observe($('race-lanes'));
  }

  fetch('assets/orl-race.json?v=20261005')
    .then(response => {
      if (!response.ok) throw new Error(response.statusText);
      return response.json();
    })
    .then(init)
    .catch(() => {
      $('race-question').textContent = 'The recorded generations could not be loaded.';
    });
})();
