# ORL project website

Project page for On-Policy Residual Learning for Diffusion Language Models.

Static HTML, CSS, and JavaScript. Served by GitHub Pages from the main branch.

The manuscript preserves anonymous authorship. The research-code link is a release placeholder.

The demo (`demo.js`) replays three recorded greedy generations of one GSM8K problem, `assets/orl-race.json`, on a
shared forward-pass axis. Sequential lanes (autoregressive, DFlash speculative decoding) list their `tokens` and,
for DFlash, the tokens each verification pass emitted (`steps`). The QwenDLM + ORL lane lists the token text of every
id it uses (`vocab`) and per 256-token block the final ids, the canvas each denoising step started from (`frames`)
and the positions not shown (`hidden`); a block costs its steps plus one cache commit. Preview locally with
`python3 -m http.server` (the data is fetched).
