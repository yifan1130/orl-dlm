'use strict';

const datasets = {
  qwen: [
    ['GSM8K', 90.4, 92.4, 2.4, 5.9],
    ['MATH-500', 88.0, 92.2, 2.1, 5.8],
    ['HumanEval', 82.9, 89.6, 1.8, 11.1],
    ['MBPP', 79.2, 79.4, 2.1, 8.0]
  ],
  llada: [
    ['GSM8K', 91.9, 92.0, 5.6, 8.5],
    ['MATH-500', 84.6, 87.0, 8.2, 10.2],
    ['HumanEval', 86.6, 89.0, 11.3, 13.1],
    ['MBPP', 77.5, 77.3, 7.7, 10.3]
  ],
  gemma: [
    ['GSM8K', 95.8, 96.0, 26.0, 30.0],
    ['MATH-500', 92.3, 92.3, 23.7, 27.4],
    ['HumanEval', 90.9, 91.9, 24.5, 27.9],
    ['MBPP', 87.7, 88.2, 23.1, 27.0]
  ]
};

function renderResults(model) {
  const data = datasets[model];
  const max = Math.max(...data.map(row => row[4])) * 1.18;
  document.querySelector('#benchmark-bars').innerHTML = data.map(([name, , , base, orl]) => `
    <div class="benchmark-chart" role="img" aria-label="${name}: base ${base.toFixed(1)}, ORL ${orl.toFixed(1)} tokens per forward, ${(orl / base).toFixed(2)} times improvement">
      <div class="bar-pair" aria-hidden="true"><div class="bar base" style="height:${base / max * 100}%"><span>${base.toFixed(1)}</span></div><div class="bar orl" style="height:${orl / max * 100}%"><span>${orl.toFixed(1)}</span></div></div>
      <div class="benchmark-name">${name}</div><span class="gain-label">${(orl / base).toFixed(2)}× TPF</span>
    </div>`).join('');
  document.querySelector('#results-panel').setAttribute('aria-labelledby', `model-${model}`);
}

const steps = [
  {
    title: '1. Generate & capture',
    description: 'The current model generates a response to the prompt. At each denoising step, the engine captures the token canvas and the exact pre-adapter residual embedding z consumed by that step, together with the rollout statistics needed for training.',
    label: 'CAPTURED STATE · SCHEMATIC TOKENIZATION',
    tokens: '<b>The</b><b>answer</b><i>[MASK]</i><i>[MASK]</i>',
    equation: 'state = (canvas, stop-gradient(z))'
  },
  {
    title: '2. Verify & retain',
    description: 'The completed response “The answer is 42” passes the verifier for 6 × 7. Its captured trajectory is eligible for training; a failed response contributes no gradient. The verified final tokens become targets for selected intermediate states.',
    label: 'VERIFIED FINAL RESPONSE · ILLUSTRATIVE',
    tokens: '<b>The</b><b>answer</b><b>is</b><b>42 ✓</b>',
    equation: 'V(prompt, response) = 1 → retain trajectory'
  },
  {
    title: '3. Pack & replay',
    description: 'Selected states are packed with reusable clean context and balanced across GPUs. The masked positions are replayed with the same captured z they consumed during rollout. Staircase attention prevents access to their own clean answer, future blocks, or unrelated noisy blocks.',
    label: 'REPLAY INPUT · SAME CAPTURED RESIDUAL',
    tokens: '<b>The</b><b>answer</b><i>[MASK]</i><i>[MASK]</i>',
    equation: 'ẽᵢ = E[xᵢ] + α gϕ(stop-gradient(zᵢ))'
  },
  {
    title: '4. Learn & refresh',
    description: 'The verified response supervises predictions at uncertain positions. AEC also penalizes insufficient contraction: with source entropy 0.40, ρ = 25%, and next entropy 0.36, the loss is (0.36 − 0.30)² = 0.0036. After the optimizer update, discard these trajectories and generate fresh ones.',
    label: 'UPDATE MODEL · DISCARD OLD TRAJECTORIES',
    tokens: '<b>verified target</b><i>+ AEC</i>',
    equation: 'fresh rollout → one update → fresh rollout'
  }
];

function renderStep(index) {
  const step = steps[index];
  document.querySelector('#step-title').textContent = step.title;
  document.querySelector('#step-description').textContent = step.description;
  document.querySelector('#state-label').textContent = step.label;
  document.querySelector('#state-tokens').innerHTML = step.tokens;
  document.querySelector('#state-equation').textContent = step.equation;
  document.querySelector('#step-panel').setAttribute('aria-labelledby', `step-tab-${index}`);
  document.querySelectorAll('.workflow-markers span').forEach((marker, i) => marker.classList.toggle('active', i === index));
}

function setupTabs(selector, onSelect) {
  const tabs = [...document.querySelectorAll(selector)];
  function select(tab) {
    tabs.forEach(item => {
      const active = item === tab;
      item.setAttribute('aria-selected', String(active));
      item.tabIndex = active ? 0 : -1;
    });
    onSelect(tab);
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        select(tabs[next]);
        tabs[next].focus();
      }
    });
  });
}
setupTabs('[data-model]', tab => renderResults(tab.dataset.model));
setupTabs('[data-step]', tab => renderStep(Number(tab.dataset.step)));
renderResults('qwen');

function updateEntropy() {
  const rho = Number(document.querySelector('#contraction').value) / 100;
  const next = Number(document.querySelector('#next-entropy').value) / 100;
  const target = (1 - rho) * 0.4;
  const shortfall = Math.max(0, next - target);
  const met = shortfall < 1e-10;
  document.querySelector('#contraction-value').textContent = `${Math.round(rho * 100)}%`;
  document.querySelector('#next-value').textContent = next.toFixed(2);
  document.querySelector('#meter-next').textContent = next.toFixed(2);
  document.querySelector('#next-bar').style.width = `${next / 0.6 * 100}%`;
  document.querySelector('#target-marker').style.left = `${target / 0.6 * 100}%`;
  document.querySelector('#target-description').textContent = `Target: ≤ ${target.toFixed(2)}`;
  document.querySelector('#loss-status').textContent = met ? 'Target met · no AEC penalty' : 'Contraction shortfall';
  document.querySelector('#loss-value').textContent = `Loss = ${(shortfall ** 2).toFixed(4)}`;
  document.querySelector('.loss-readout').classList.toggle('met', met);
}
document.querySelectorAll('.entropy-demo input').forEach(input => input.addEventListener('input', updateEntropy));
updateEntropy();

const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('#nav-links');
function closeMenu() {
  menu.setAttribute('aria-expanded', 'false');
  menu.setAttribute('aria-label', 'Open navigation');
  nav.classList.remove('open');
}
menu.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  nav.classList.toggle('open', open);
});
nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });
document.addEventListener('click', event => { if (!event.target.closest('.nav-shell')) closeMenu(); });

const observed = [...document.querySelectorAll('main section[id], #motivation')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('#nav-links a, .page-outline a').forEach(link => {
          const active = link.hash === `#${entry.target.id}`;
          link.classList.toggle('active', active);
          if (active) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      }
    });
  }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
  observed.forEach(section => observer.observe(section));
}

const dialog = document.querySelector('#figure-dialog');
const figureNames = { 'training-loop': 'Figure 9 · The ORL training loop', overview: 'Figure 1 · Residual learning and the speed–quality frontier', systems: 'Figure 3 · Cumulative system optimizations', motivation: 'Figure 2 · Three observations behind ORL', 'data-efficiency': 'Figure 5 · Post-training data efficiency', stability: 'Figure 6 · Stable residual optimization' };
document.querySelectorAll('[data-zoom]').forEach(button => {
  button.addEventListener('click', () => {
    const source = button.querySelector('img');
    const target = document.querySelector('#dialog-image');
    target.src = source.src;
    target.alt = source.alt;
    document.querySelector('#dialog-caption').textContent = figureNames[button.dataset.zoom];
    dialog.showModal();
    document.body.style.overflow = 'hidden';
  });
});
document.querySelector('#close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) { const bounds = dialog.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close(); } });
dialog.addEventListener('close', () => { document.body.style.overflow = ''; });

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const text = document.querySelector('#bibtex').textContent;
  const button = document.querySelector('#copy-citation');
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const field = document.createElement('textarea');
      field.value = text;
      field.style.position = 'fixed';
      field.style.opacity = '0';
      document.body.append(field);
      field.select();
      const copied = document.execCommand('copy');
      field.remove();
      if (!copied) throw new Error('Copy unavailable');
    }
    button.textContent = 'Copied ✓';
    document.querySelector('#copy-status').textContent = 'Citation copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    button.textContent = 'Select & copy';
    document.querySelector('#copy-status').textContent = 'Copy unavailable. Citation selected; use your browser’s copy command.';
  }
  setTimeout(() => { button.textContent = 'Copy ⧉'; }, 2500);
});
