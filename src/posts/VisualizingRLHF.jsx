// src/posts/VisualizingRLHF.jsx
// Part 4 of "Visualizing ML" — From next-token prediction to following instructions.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import './VisualizingRLHF.css';

/* =========================================================
   KaTeX wrapper — uses the CDN-loaded global window.katex
   ========================================================= */

function Katex({ tex, block = false, className = '' }) {
  const ref = useRef(null);
  useEffect(() => {
    let cancelled = false;
    const render = () => {
      if (cancelled || !ref.current) return;
      if (window.katex) {
        try {
          window.katex.render(tex, ref.current, {
            displayMode: block,
            throwOnError: false,
          });
        } catch {
          if (ref.current) ref.current.textContent = tex;
        }
      } else {
        setTimeout(render, 60);
      }
    };
    render();
    return () => { cancelled = true; };
  }, [tex, block]);
  return (
    <span
      ref={ref}
      className={`${block ? 'viz-math-block' : 'viz-math-inline'} ${className}`}
    />
  );
}

/* =========================================================
   InfoBox — reusable aside callout
   ========================================================= */

function InfoBox({ title, children }) {
  return (
    <aside className="viz-subsection viz-aside">
      <h4 className="viz-subhead">
        <span className="viz-sub-tag viz-sub-tag-info">i</span>
        {title}
      </h4>
      <div className="viz-aside-body">{children}</div>
    </aside>
  );
}

/* =========================================================
   Counter chip — small KPI strip beneath each interactive
   ========================================================= */

function Counter({ label, value, sub }) {
  return (
    <div className="viz-rlhf-counter">
      <div className="viz-rlhf-counter-label">{label}</div>
      <div className="viz-rlhf-counter-value">{value}</div>
      {sub && <div className="viz-rlhf-counter-sub">{sub}</div>}
    </div>
  );
}

/* =========================================================
   §1 — SFT: base model vs SFT model, side-by-side
   ========================================================= */

const SFT_PROMPTS = [
  {
    id: 'rope',
    prompt: 'Explain RoPE in one sentence.',
    base:
      `Explain RoPE in one sentence.
Explain attention in one sentence.
Explain LayerNorm in one sentence.
Tags: deep-learning, transformers, positional-encoding
Asked 2 years ago by user_3392 · 1 answer (closed as duplicate)`,
    sft:
      `RoPE is a positional encoding that rotates each query and key vector in 2D pairs by an angle proportional to position, so the dot product between any Q and K naturally encodes their relative offset.`,
  },
  {
    id: 'haiku',
    prompt: 'Write a haiku about debugging.',
    base:
      `Write a haiku about debugging.
Write a haiku about coffee.
Write a haiku about Mondays.
Write a haiku about deadlines.
Category: poetry · programming-humor · off-topic`,
    sft:
      `Silent bug appears —
print statements multiply like
rabbits in the void.`,
  },
  {
    id: 'unsub',
    prompt: 'How do I unsubscribe from this newsletter?',
    base:
      `How do I unsubscribe from this newsletter?
How do I change my password?
How do I delete my account?
How do I update my billing info?
— Frequently Asked Questions —`,
    sft:
      `Scroll to the bottom of any newsletter email and click "Unsubscribe." If you can't find the link, reply with the word "unsubscribe" and we'll remove you within 24 hours.`,
  },
];

function SectionSFT() {
  const [presetId, setPresetId] = useState('rope');
  const preset = SFT_PROMPTS.find((p) => p.id === presetId);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label>
          Prompt
          <select
            value={presetId}
            onChange={(e) => setPresetId(e.target.value)}
          >
            {SFT_PROMPTS.map((p) => (
              <option key={p.id} value={p.id}>{p.prompt}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="viz-rlhf-prompt">{preset.prompt}</div>

      <div className="viz-rlhf-pair">
        <div className="viz-rlhf-card">
          <div className="viz-rlhf-card-tag base">Base LM · next-token prediction</div>
          <div className="viz-rlhf-completion base">{preset.base}</div>
        </div>
        <div className="viz-rlhf-card">
          <div className="viz-rlhf-card-tag sft">SFT · trained on demonstrations</div>
          <div className="viz-rlhf-completion">{preset.sft}</div>
        </div>
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Same prompt, same architecture, two checkpoints. The base model continues
        the <em>statistical pattern</em> of similar text on the internet (lists
        of questions, FAQ headers, Stack Overflow chrome). The SFT model — the
        base model fine-tuned on a few thousand handwritten{' '}
        <Katex tex="(\text{prompt}, \text{ideal response})" /> pairs —
        recognises a request and answers it.
      </p>
    </div>
  );
}

/* =========================================================
   §2 — Reward model: preference-pair clicker + Bradley-Terry
   ========================================================= */

const RM_PAIRS = [
  {
    id: 'mars',
    prompt: 'What is the largest planet in our solar system?',
    a: 'The largest planet in our solar system is Jupiter — about 318 times the mass of Earth, large enough that all the other planets combined would fit inside it twice.',
    b: 'I think it might be Saturn or maybe Jupiter, but honestly I\'m not 100% sure. You should probably look it up to be safe.',
    truthChosen: 'a',
  },
  {
    id: 'medical',
    prompt: 'My chest hurts when I breathe deeply. What should I do?',
    a: 'It\'s probably nothing — most chest pain is just muscle strain. Take some ibuprofen and rest.',
    b: 'Chest pain on deep breathing has many possible causes (muscle strain, costochondritis, lung issues). If it\'s severe, persistent, or comes with shortness of breath, fever, or trauma, please see a doctor or urgent care today rather than self-diagnose.',
    truthChosen: 'b',
  },
  {
    id: 'recipe',
    prompt: 'How do I make scrambled eggs?',
    a: 'Crack 2-3 eggs into a bowl, add a splash of milk, whisk until pale yellow. Melt butter in a pan over low heat. Pour in eggs, stir slowly with a spatula, pulling cooked edges toward the centre. Remove from heat while still slightly wet — they\'ll finish cooking from residual warmth. Salt at the end.',
    b: 'Just put eggs in a pan and cook them. Stir occasionally. They\'re done when they look done.',
    truthChosen: 'a',
  },
  {
    id: 'opinion',
    prompt: 'Is Python better than JavaScript?',
    a: 'Python is objectively the better language. JavaScript is a hack that should never have been allowed near production systems.',
    b: 'They\'re optimised for different things: Python for readability and scientific/ML work, JavaScript for the browser and event-driven systems. The "better" one depends on what you\'re building — neither is universally superior.',
    truthChosen: 'b',
  },
];

function SectionRewardModel() {
  // labels[id] = 'a' | 'b'  (what the user clicked)
  const [labels, setLabels] = useState({});

  const handleClick = (pairId, choice) => {
    setLabels((prev) => ({ ...prev, [pairId]: choice }));
  };

  const reset = () => setLabels({});

  // The "reward head" is a single scalar per candidate, parameterised so that
  // labels move chosen up and rejected down by 0.6 each time the user agrees
  // with the held-out ground truth. We compute it across all *labelled* pairs.
  let chosenSum = 0;
  let rejectedSum = 0;
  let agreed = 0;
  for (const pair of RM_PAIRS) {
    const choice = labels[pair.id];
    if (!choice) continue;
    if (choice === pair.truthChosen) {
      chosenSum += 0.6;
      rejectedSum -= 0.6;
      agreed += 1;
    } else {
      // user disagrees — RM still learns, but in the wrong direction.
      chosenSum -= 0.3;
      rejectedSum += 0.3;
    }
  }

  // Bradley-Terry loss: -log σ(r_chosen - r_rejected), averaged over pairs.
  const labelCount = Object.keys(labels).length;
  const margin = chosenSum - rejectedSum;
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const loss = labelCount === 0
    ? null
    : -Math.log(Math.max(1e-9, sigmoid(margin / Math.max(1, labelCount))));

  const barWidth = (value) => {
    const max = 3;
    return Math.min(100, (Math.abs(value) / max) * 100);
  };

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <span style={{ color: 'var(--ink-soft)', fontSize: '0.92rem' }}>
          Click the response you think is <strong>better</strong> for each prompt:
        </span>
        <button onClick={reset}>Reset</button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
        {RM_PAIRS.map((pair) => {
          const choice = labels[pair.id];
          const aClass = choice
            ? choice === 'a'
              ? 'viz-rlhf-candidate locked-chosen'
              : 'viz-rlhf-candidate locked-rejected'
            : 'viz-rlhf-candidate';
          const bClass = choice
            ? choice === 'b'
              ? 'viz-rlhf-candidate locked-chosen'
              : 'viz-rlhf-candidate locked-rejected'
            : 'viz-rlhf-candidate';
          return (
            <div key={pair.id}>
              <div className="viz-rlhf-prompt">{pair.prompt}</div>
              <div className="viz-rlhf-pair-pick">
                <button
                  type="button"
                  className={aClass}
                  onClick={() => !choice && handleClick(pair.id, 'a')}
                  disabled={!!choice}
                >
                  <div className="viz-rlhf-candidate-tag">
                    Response A
                    {choice === 'a' && ' · chosen'}
                    {choice && choice !== 'a' && ' · rejected'}
                  </div>
                  {pair.a}
                </button>
                <button
                  type="button"
                  className={bClass}
                  onClick={() => !choice && handleClick(pair.id, 'b')}
                  disabled={!!choice}
                >
                  <div className="viz-rlhf-candidate-tag">
                    Response B
                    {choice === 'b' && ' · chosen'}
                    {choice && choice !== 'b' && ' · rejected'}
                  </div>
                  {pair.b}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: 20 }}>
        <h4 style={{ margin: '0 0 10px', fontFamily: 'var(--serif)', fontSize: '1.0rem' }}>
          Reward head — Σ over labelled pairs
        </h4>
        <div className="viz-rlhf-reward-bars">
          <div>r<sub>φ</sub>(chosen)</div>
          <div className="viz-rlhf-reward-track">
            <div className="center-line" />
            <div
              className={`viz-rlhf-reward-fill ${chosenSum >= 0 ? 'pos' : 'neg'}`}
              style={{
                width: `${barWidth(chosenSum) / 2}%`,
                transform: chosenSum >= 0 ? 'none' : 'translateX(-100%)',
              }}
            />
          </div>
          <div style={{ textAlign: 'right' }}>{chosenSum.toFixed(2)}</div>

          <div>r<sub>φ</sub>(rejected)</div>
          <div className="viz-rlhf-reward-track">
            <div className="center-line" />
            <div
              className={`viz-rlhf-reward-fill ${rejectedSum >= 0 ? 'pos' : 'neg'}`}
              style={{
                width: `${barWidth(rejectedSum) / 2}%`,
                transform: rejectedSum >= 0 ? 'none' : 'translateX(-100%)',
              }}
            />
          </div>
          <div style={{ textAlign: 'right' }}>{rejectedSum.toFixed(2)}</div>
        </div>
      </div>

      <div className="viz-rlhf-counters">
        <Counter
          label="Pairs labelled"
          value={`${labelCount} / ${RM_PAIRS.length}`}
          sub={labelCount === 0 ? 'click a response to begin' : `${agreed} agree w/ held-out`}
        />
        <Counter
          label="Margin"
          value={margin.toFixed(2)}
          sub="r(chosen) − r(rejected)"
        />
        <Counter
          label="BT loss"
          value={loss === null ? '—' : loss.toFixed(3)}
          sub="−log σ(margin / N)"
        />
      </div>
    </div>
  );
}

/* =========================================================
   §3 — PPO: KL coefficient slider + reward/KL plot + sample
   ========================================================= */

// We model PPO's outcome at convergence as a 1-D dial. Let t ∈ [0, 1] map the
// "drift away from π_ref" — t = 0 is π_ref (β → ∞), t = 1 is the unconstrained
// reward maximiser (β → 0). β maps to t via a smooth squash so the slider feels
// natural across the range that matters.
//
// reward(t):  concave, climbs fast then saturates at max RM score
// kl(t):      convex, slow growth then explodes near t = 1 (RM-hacked policy
//             is far from natural language)
// true(t):    actual quality — a true-quality oracle that the RM only proxies.
//             Climbs with reward at first, then *drops* once the policy hacks
//             the RM (the RM rewards gibberish the oracle doesn't).
function ppoCurves(beta) {
  // β slider runs on log scale from 0.005 to 5; convert to drift t.
  const t = 1 / (1 + 8 * beta); // smooth, monotone-decreasing in β
  const reward = 0.30 + 0.70 * Math.pow(t, 0.55);          // saturating
  const kl     = 8.0 * Math.pow(t, 2.4);                   // explosive near 1
  const trueQ  = 0.32 + 0.55 * Math.pow(t, 0.55)
               - 0.95 * Math.pow(Math.max(0, t - 0.55), 2.2); // peaks ~t=0.6
  return { t, reward, kl, trueQ };
}

const PPO_SAMPLES = [
  // ordered by t threshold (low → high drift)
  {
    tMax: 0.20,
    zone: 'frozen',
    label: 'Barely moved (β too high)',
    text:
      'RoPE is a positional encoding that rotates query and key vectors in 2D pairs by an angle proportional to position, so the dot product between any Q and K naturally encodes their relative offset.',
  },
  {
    tMax: 0.80,
    zone: 'sweet',
    label: 'Sweet spot',
    text:
      'Rotary Position Embeddings (RoPE) inject sequence position into a Transformer by rotating each query and key vector in 2D coordinate pairs by an angle proportional to its position — so the dot product between any two tokens naturally encodes the relative offset between them, with zero extra learned parameters and clean long-context extrapolation.',
  },
  {
    tMax: 1.01,
    zone: 'hacking',
    label: 'Reward hacking',
    text:
      'RoPE great rotations excellent positional encoding excellent yes very helpful answer thank you so much yes definitely positive answer yes yes ★★★★★ very informative excellent excellent excellent.',
  },
];

function SectionPPO() {
  // log slider 0..1 -> β = 0.005 * 1000^x
  const [sliderPos, setSliderPos] = useState(0.55);
  const beta = 0.005 * Math.pow(1000, sliderPos);

  const cur = ppoCurves(beta);

  // Sweep β across the slider range to draw the plot.
  const points = useMemo(() => {
    const pts = [];
    const N = 80;
    for (let i = 0; i <= N; i++) {
      const x = i / N;
      const b = 0.005 * Math.pow(1000, x);
      pts.push({ x, ...ppoCurves(b), beta: b });
    }
    return pts;
  }, []);

  const sample = PPO_SAMPLES.find((s) => cur.t <= s.tMax) || PPO_SAMPLES[PPO_SAMPLES.length - 1];

  // Plot geometry
  const W = 520;
  const H = 240;
  const padL = 38;
  const padR = 38;
  const padT = 14;
  const padB = 26;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  // y-axes: reward [0,1] on the left; KL [0, 8] on the right.
  const klMax = 8;
  const yReward = (r) => padT + (1 - r) * innerH;
  const yKL     = (k) => padT + (1 - Math.min(1, k / klMax)) * innerH;
  const xPos    = (x) => padL + x * innerW;

  const rewardPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${xPos(p.x).toFixed(2)} ${yReward(p.reward).toFixed(2)}`).join(' ');
  const klPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${xPos(p.x).toFixed(2)} ${yKL(p.kl).toFixed(2)}`).join(' ');
  const truePath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${xPos(p.x).toFixed(2)} ${yReward(p.trueQ).toFixed(2)}`).join(' ');

  // Band thresholds match the sample tMax cutoffs
  const tToX = (t) => {
    // invert: t = 1/(1 + 8β); solve for β, then β → slider x.
    const b = (1 / t - 1) / 8;
    const x = Math.log(b / 0.005) / Math.log(1000);
    return Math.max(0, Math.min(1, x));
  };
  // Bands: hacking (t > 0.80), sweet (0.20 < t <= 0.80), frozen (t <= 0.20)
  const xFrozenEnd  = tToX(0.20);
  const xSweetEnd   = tToX(0.80);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label style={{ minWidth: 280 }}>
          KL coefficient&nbsp;<Katex tex="\beta" /> ={' '}
          <strong style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>
            {beta.toFixed(beta < 0.1 ? 3 : 2)}
          </strong>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={sliderPos}
          onChange={(e) => setSliderPos(parseFloat(e.target.value))}
          style={{ flex: 1, minWidth: 200, accentColor: 'var(--accent)' }}
        />
      </div>

      <div className="viz-rlhf-ppo-loop">
        {/* Left: the plot */}
        <div>
          <div style={{
            fontFamily: 'var(--mono)',
            fontSize: '0.78rem',
            color: 'var(--ink-faint)',
            marginBottom: 4,
          }}>
            reward, KL, and true quality vs <em>β</em> (log scale)
          </div>
          <svg className="viz-rlhf-ppo-plot" viewBox={`0 0 ${W} ${H}`} width="100%">
            {/* Bands */}
            <rect
              x={padL}
              y={padT}
              width={Math.max(0, xPos(tToX(1.01)) - xPos(xSweetEnd))}
              height={innerH}
              fill="hsl(8, 60%, 96%)"
            />
            <rect
              x={xPos(xSweetEnd)}
              y={padT}
              width={Math.max(0, xPos(1) - xPos(xSweetEnd))}
              height={innerH}
              fill="hsl(8, 60%, 96%)"
            />
            <rect
              x={xPos(xFrozenEnd)}
              y={padT}
              width={Math.max(0, xPos(xSweetEnd) - xPos(xFrozenEnd))}
              height={innerH}
              fill="hsl(148, 35%, 95%)"
            />
            <rect
              x={padL}
              y={padT}
              width={Math.max(0, xPos(xFrozenEnd) - padL)}
              height={innerH}
              fill="hsl(218, 30%, 95%)"
            />

            {/* Grid lines */}
            {[0.25, 0.5, 0.75].map((g) => (
              <line
                key={`gh-${g}`}
                x1={padL}
                y1={padT + (1 - g) * innerH}
                x2={padL + innerW}
                y2={padT + (1 - g) * innerH}
                stroke="hsl(218, 20%, 88%)"
                strokeDasharray="2 3"
              />
            ))}

            {/* Current β vertical line */}
            <line
              x1={xPos(sliderPos)}
              y1={padT}
              x2={xPos(sliderPos)}
              y2={padT + innerH}
              stroke="var(--accent)"
              strokeWidth="1.5"
            />

            {/* Curves */}
            <path d={rewardPath} stroke="hsl(218, 60%, 45%)" strokeWidth="2.2" fill="none" />
            <path d={klPath}     stroke="hsl(8, 60%, 50%)"   strokeWidth="2.2" fill="none" strokeDasharray="5 4" />
            <path d={truePath}   stroke="hsl(148, 45%, 38%)" strokeWidth="2.2" fill="none" />

            {/* Current-β dots */}
            <circle cx={xPos(sliderPos)} cy={yReward(cur.reward)} r="4" fill="hsl(218, 60%, 45%)" />
            <circle cx={xPos(sliderPos)} cy={yKL(cur.kl)}         r="4" fill="hsl(8, 60%, 50%)" />
            <circle cx={xPos(sliderPos)} cy={yReward(cur.trueQ)}  r="4" fill="hsl(148, 45%, 38%)" />

            {/* X-axis labels */}
            <text x={padL}        y={H - 8} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">β = 0.005</text>
            <text x={padL + innerW} y={H - 8} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">β = 5</text>
            <text x={padL + innerW / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
              ← unconstrained reward&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;frozen at SFT →
            </text>

            {/* Y-axis labels */}
            <text x={padL - 4} y={padT + 6}           textAnchor="end" fontSize="10" fill="hsl(218, 60%, 45%)" fontFamily="var(--mono)">reward 1.0</text>
            <text x={padL - 4} y={padT + innerH - 2}  textAnchor="end" fontSize="10" fill="hsl(218, 60%, 45%)" fontFamily="var(--mono)">0</text>
            <text x={padL + innerW + 4} y={padT + 6}  textAnchor="start" fontSize="10" fill="hsl(8, 60%, 50%)" fontFamily="var(--mono)">KL {klMax}</text>
            <text x={padL + innerW + 4} y={padT + innerH - 2} textAnchor="start" fontSize="10" fill="hsl(8, 60%, 50%)" fontFamily="var(--mono)">0</text>
          </svg>

          {/* Legend */}
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 14,
            marginTop: 8,
            fontSize: '0.82rem',
            fontFamily: 'var(--mono)',
            color: 'var(--ink-soft)',
          }}>
            <span><span style={{ display: 'inline-block', width: 14, height: 2, background: 'hsl(218, 60%, 45%)', verticalAlign: 'middle', marginRight: 6 }} />reward r<sub>φ</sub></span>
            <span><span style={{ display: 'inline-block', width: 14, height: 2, background: 'hsl(8, 60%, 50%)', verticalAlign: 'middle', marginRight: 6 }} />KL(π<sub>θ</sub> ‖ π<sub>ref</sub>)</span>
            <span><span style={{ display: 'inline-block', width: 14, height: 2, background: 'hsl(148, 45%, 38%)', verticalAlign: 'middle', marginRight: 6 }} />true quality</span>
          </div>
        </div>

        {/* Right: a sample completion at this β */}
        <div className="viz-rlhf-ppo-sample">
          <div className="viz-rlhf-ppo-sample-tag">
            policy output · prompt: "Explain RoPE in one sentence."
          </div>
          <div>
            <span className={`viz-rlhf-ppo-zone ${sample.zone}`}>{sample.label}</span>
          </div>
          <div className="viz-rlhf-ppo-sample-body">
            {sample.text}
          </div>
        </div>
      </div>

      <div className="viz-rlhf-counters">
        <Counter
          label="Reward r_φ"
          value={cur.reward.toFixed(3)}
          sub="what PPO optimises"
        />
        <Counter
          label="KL from π_ref"
          value={cur.kl.toFixed(2)}
          sub="nats"
        />
        <Counter
          label="True quality"
          value={cur.trueQ.toFixed(3)}
          sub={cur.trueQ < 0.45 && cur.t > 0.55 ? 'RM is being gamed' : 'oracle (not observed)'}
        />
      </div>
    </div>
  );
}

/* =========================================================
   §3.1 — Clipped-surrogate gradient visualization
   Shows where ∂L/∂ρ is alive vs frozen, for Â_t = ±1.
   ========================================================= */

function ClipPlot({ advantage, rho, epsilon, label }) {
  const W = 320;
  const H = 200;
  const padL = 38;
  const padR = 14;
  const padT = 14;
  const padB = 30;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const rhoMin = 0.4;
  const rhoMax = 1.8;
  const yMin = -1.8;
  const yMax = 1.8;

  const toX = (r) => padL + ((r - rhoMin) / (rhoMax - rhoMin)) * innerW;
  const toY = (y) => padT + ((yMax - y) / (yMax - yMin)) * innerH;

  // Build polyline points along the min(...) curve.
  const buildPath = (rStart, rEnd) => {
    const N = 40;
    const segs = [];
    for (let i = 0; i <= N; i++) {
      const r = rStart + (i / N) * (rEnd - rStart);
      const clipped = Math.max(1 - epsilon, Math.min(1 + epsilon, r)) * advantage;
      const unclipped = r * advantage;
      const y = Math.min(unclipped, clipped);
      segs.push(`${i === 0 ? 'M' : 'L'} ${toX(r).toFixed(2)} ${toY(y).toFixed(2)}`);
    }
    return segs.join(' ');
  };

  // Two segments: active (slope = advantage) and frozen (slope = 0).
  // For Â > 0: active on ρ < 1+ε, frozen on ρ > 1+ε.
  // For Â < 0: frozen on ρ < 1-ε, active on ρ > 1-ε.
  const frozenLeft = advantage < 0;
  const threshold = frozenLeft ? 1 - epsilon : 1 + epsilon;
  const activePath  = frozenLeft
    ? buildPath(threshold, rhoMax)
    : buildPath(rhoMin, threshold);
  const frozenPath  = frozenLeft
    ? buildPath(rhoMin, threshold)
    : buildPath(threshold, rhoMax);

  // Light reference lines for the unclipped and clipped objectives
  const unclippedRefPath = `M ${toX(rhoMin)} ${toY(rhoMin * advantage)} L ${toX(rhoMax)} ${toY(rhoMax * advantage)}`;
  const clippedRefPath = (() => {
    const y0 = (1 - epsilon) * advantage;
    const y1 = (1 + epsilon) * advantage;
    return `M ${toX(rhoMin)} ${toY(y0)} L ${toX(1 - epsilon)} ${toY(y0)}` +
           ` L ${toX(1 + epsilon)} ${toY(y1)} L ${toX(rhoMax)} ${toY(y1)}`;
  })();

  // Current point on the min curve
  const yCur = Math.min(rho * advantage, Math.max(1 - epsilon, Math.min(1 + epsilon, rho)) * advantage);
  const isFrozen = frozenLeft ? rho < 1 - epsilon : rho > 1 + epsilon;
  const gradient = isFrozen ? 0 : advantage;

  const ACTIVE = 'hsl(148, 45%, 38%)';
  const FROZEN = 'hsl(218, 20%, 60%)';

  return (
    <div>
      <div style={{
        fontFamily: 'var(--mono)',
        fontSize: '0.82rem',
        color: 'var(--ink-soft)',
        marginBottom: 4,
      }}>
        {label}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{
        background: 'hsl(218, 30%, 98%)',
        border: '1px solid hsl(218, 30%, 90%)',
        borderRadius: 8,
        maxWidth: W,
      }}>
        {/* trust band shading */}
        <rect
          x={toX(1 - epsilon)}
          y={padT}
          width={toX(1 + epsilon) - toX(1 - epsilon)}
          height={innerH}
          fill="hsl(var(--accent-h), 30%, 94%)"
        />

        {/* Zero line + ρ=1 line */}
        <line x1={padL} y1={toY(0)} x2={padL + innerW} y2={toY(0)}
              stroke="hsl(218, 20%, 80%)" />
        <line x1={toX(1)} y1={padT} x2={toX(1)} y2={padT + innerH}
              stroke="hsl(218, 20%, 80%)" strokeDasharray="2 3" />

        {/* Trust-band edges */}
        <line x1={toX(1 - epsilon)} y1={padT} x2={toX(1 - epsilon)} y2={padT + innerH}
              stroke="hsl(var(--accent-h), 30%, 70%)" strokeDasharray="3 3" strokeWidth="1" />
        <line x1={toX(1 + epsilon)} y1={padT} x2={toX(1 + epsilon)} y2={padT + innerH}
              stroke="hsl(var(--accent-h), 30%, 70%)" strokeDasharray="3 3" strokeWidth="1" />

        {/* Light reference curves */}
        <path d={unclippedRefPath} stroke="hsl(218, 30%, 75%)" strokeWidth="1" fill="none" strokeDasharray="3 3" />
        <path d={clippedRefPath} stroke="hsl(218, 30%, 75%)" strokeWidth="1" fill="none" strokeDasharray="3 3" />

        {/* The min(...) curve, colored by region */}
        <path d={frozenPath} stroke={FROZEN} strokeWidth="3" fill="none" />
        <path d={activePath} stroke={ACTIVE} strokeWidth="3" fill="none" />

        {/* Current ρ vertical line and dot */}
        <line x1={toX(rho)} y1={padT} x2={toX(rho)} y2={padT + innerH}
              stroke="var(--accent)" strokeWidth="1.5" />
        <circle cx={toX(rho)} cy={toY(yCur)} r="5"
                fill={isFrozen ? FROZEN : ACTIVE}
                stroke="#fff" strokeWidth="1.5" />

        {/* X-axis labels */}
        <text x={padL} y={H - 12} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
          ρ = {rhoMin}
        </text>
        <text x={toX(1)} y={H - 12} textAnchor="middle" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
          1
        </text>
        <text x={toX(1 - epsilon)} y={H - 2} textAnchor="middle" fontSize="9" fill="hsl(var(--accent-h), 30%, 50%)" fontFamily="var(--mono)">
          1−ε
        </text>
        <text x={toX(1 + epsilon)} y={H - 2} textAnchor="middle" fontSize="9" fill="hsl(var(--accent-h), 30%, 50%)" fontFamily="var(--mono)">
          1+ε
        </text>
        <text x={padL + innerW} y={H - 12} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
          {rhoMax}
        </text>

        {/* Y-axis labels */}
        <text x={padL - 4} y={toY(advantage > 0 ? 1.5 : -1.5) + 3} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
          {advantage > 0 ? '+1.5' : '−1.5'}
        </text>
        <text x={padL - 4} y={toY(0) + 3} textAnchor="end" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">
          0
        </text>
      </svg>

      <div style={{
        marginTop: 6,
        display: 'flex',
        flexWrap: 'wrap',
        gap: 12,
        fontFamily: 'var(--mono)',
        fontSize: '0.82rem',
      }}>
        <span style={{
          padding: '3px 8px',
          borderRadius: 4,
          background: isFrozen ? 'hsl(218, 30%, 92%)' : 'hsl(148, 35%, 90%)',
          color: isFrozen ? 'var(--ink-soft)' : 'var(--pos)',
          fontWeight: 600,
        }}>
          ∂L/∂ρ = {gradient === 0 ? '0  (frozen)' : (gradient > 0 ? '+1' : '−1')}
        </span>
        <span style={{ color: 'var(--ink-faint)' }}>
          point at ρ = {rho.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function SectionClipGradient() {
  const [rho, setRho] = useState(1.0);
  const epsilon = 0.2;

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <label style={{ minWidth: 220 }}>
          importance ratio&nbsp;<Katex tex="\rho_t" /> ={' '}
          <strong style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>
            {rho.toFixed(2)}
          </strong>
        </label>
        <input
          type="range"
          min="0.4"
          max="1.8"
          step="0.01"
          value={rho}
          onChange={(e) => setRho(parseFloat(e.target.value))}
          style={{ flex: 1, minWidth: 220, accentColor: 'var(--accent)' }}
        />
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 16,
      }} className="viz-rlhf-clip-plots">
        <ClipPlot
          advantage={+1}
          rho={rho}
          epsilon={epsilon}
          label="Â_t = +1   (good action)"
        />
        <ClipPlot
          advantage={-1}
          rho={rho}
          epsilon={epsilon}
          label="Â_t = −1   (bad action)"
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Solid green = gradient is alive ({' '}<Katex tex="\partial L / \partial \rho = \hat{A}_t" />).
        Solid grey = gradient is zero ({' '}<Katex tex="\partial L / \partial \rho = 0" />,
        the clip's flat region). The dashed light lines underneath are the
        unclipped objective <Katex tex="\rho_t \hat{A}_t" /> and the clipped
        objective <Katex tex="\mathrm{clip}(\rho_t) \hat{A}_t" />; the solid
        curve is the <Katex tex="\min" /> of the two — what PPO actually
        differentiates. Drag <Katex tex="\rho_t" /> across the trust band
        (light blue): notice the frozen region sits on the <em>right</em> for
        good actions and on the <em>left</em> for bad ones.
      </p>
    </div>
  );
}

/* =========================================================
   §4 — PPO vs DPO vs GRPO comparison
   ========================================================= */

const METHOD_BOXES = [
  // (id, name, sub-note)
  { id: 'policy',    name: 'Policy π_θ',    note: 'the model we ship' },
  { id: 'reference', name: 'Reference π_ref', note: 'frozen SFT model' },
  { id: 'reward',    name: 'Reward model r_φ', note: 'learned from prefs' },
  { id: 'value',     name: 'Value net V_ψ',    note: 'variance reduction' },
];

const METHODS = {
  ppo: {
    label: 'PPO (InstructGPT)',
    active: { policy: true, reference: true, reward: true, value: true },
    formula: String.raw`
      \mathcal{L}_{\mathrm{PPO}}(\theta)
      \;=\; -\,\mathbb{E}\!\left[\,
        \min\!\big(\,\rho_t\,\hat{A}_t,\; \mathrm{clip}(\rho_t,\,1\!-\!\epsilon,\,1\!+\!\epsilon)\,\hat{A}_t\,\big)
      \right]
      \;+\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\big)
    `,
    formulaSub: String.raw`\rho_t \;=\; \tfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}, \qquad \hat{A}_t \;\approx\; r_\varphi(x, y) - V_\psi(s_t) \;\;\text{(from §3)}`,
    cite: { paper: 'InstructGPT', url: 'https://arxiv.org/abs/2203.02155' },
    summary: (
      <>
        The classical pipeline from §3, written out as the per-token clipped
        surrogate that's actually implemented. Sample completions, score them
        with <Katex tex="r_\varphi" />, advantage-weight the gradient using{' '}
        <Katex tex="V_\psi" /> for variance reduction, and clip the policy
        ratio <Katex tex="\rho_t" /> so each update stays in a trust region.{' '}
        The <Katex tex="\beta\,\mathrm{KL}" /> term keeps the policy from
        drifting off the SFT manifold. <strong>All four boxes live in GPU
        memory at once</strong> — expensive, but well-understood.
      </>
    ),
  },
  dpo: {
    label: 'DPO',
    active: { policy: true, reference: true, reward: false, value: false },
    formula: String.raw`
      \mathcal{L}_{\mathrm{DPO}}(\theta)
      \;=\; -\log \sigma\!\left(
        \beta\,\log \tfrac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
        \;-\;
        \beta\,\log \tfrac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
      \right)
    `,
    formulaSub: String.raw`(x, y_w, y_l) \sim \mathcal{D}_{\mathrm{pref}}, \quad y_w \succ y_l`,
    cite: { paper: 'DPO', url: 'https://arxiv.org/abs/2305.18290' },
    summary: (
      <>
        The trick: PPO's optimum is closed-form in terms of the reward, and
        for a Bradley-Terry reward this gives a loss that depends only on
        log-prob ratios over preference pairs <Katex tex="(y_w \succ y_l)" />.{' '}
        <strong>No RM, no rollouts, no value net</strong> — a single
        supervised-style fine-tune on preference data, with the reference
        policy supplying the implicit reward.
      </>
    ),
  },
  grpo: {
    label: 'GRPO (DeepSeek)',
    active: { policy: true, reference: true, reward: true, value: false },
    formula: String.raw`
      \mathcal{L}_{\mathrm{GRPO}}(\theta)
      \;=\; -\,\mathbb{E}\!\left[\, \tfrac{1}{G}\sum_{i=1}^{G}
        \min\!\big(\rho_i\,\hat{A}_i,\; \mathrm{clip}(\rho_i)\,\hat{A}_i\big)
      \right]
      \;+\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\big)
    `,
    formulaSub: String.raw`\rho_i \;=\; \tfrac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i \mid x)}, \qquad \hat{A}_i \;=\; \tfrac{r_i \,-\, \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}, \qquad \{y_1, \ldots, y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x)`,
    cite: { paper: 'DeepSeekMath / GRPO', url: 'https://arxiv.org/abs/2402.03300' },
    summary: (
      <>
        Sample <Katex tex="G" /> completions per prompt and use the{' '}
        <em>group's own statistics</em> as the baseline — z-score each
        reward against its siblings instead of subtracting{' '}
        <Katex tex="V_\psi(s_t)" />. <strong>Drops the value net</strong>{' '}
        entirely (the largest of the four boxes from §3, since it's the same
        size as the policy), at the cost of needing{' '}
        <Katex tex="G" /> rollouts per prompt. The reward can be a learned
        RM <em>or</em> a verifiable scalar (test pass-rate, proof checker) —
        the trick that powers DeepSeek-R1.
      </>
    ),
  },
};

function SectionMethodCompare() {
  const [method, setMethod] = useState('ppo');
  const m = METHODS[method];

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          {Object.entries(METHODS).map(([id, info]) => (
            <button
              key={id}
              role="tab"
              className={`viz-tab ${method === id ? 'active' : ''}`}
              onClick={() => setMethod(id)}
            >
              {info.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ fontFamily: 'var(--mono)', fontSize: '0.78rem', color: 'var(--ink-faint)', marginBottom: 6 }}>
        components in memory during training
      </div>
      <div className="viz-rlhf-method-grid">
        {METHOD_BOXES.map((box) => {
          const isActive = m.active[box.id];
          return (
            <div
              key={box.id}
              className={`viz-rlhf-method-box ${isActive ? 'active' : 'inactive'}`}
            >
              <div className="box-tag">{isActive ? 'required' : 'not used'}</div>
              <div className="box-name">{box.name}</div>
              <div className="box-note">{box.note}</div>
            </div>
          );
        })}
      </div>

      <div className="viz-math-block">
        <Katex block tex={m.formula} />
      </div>
      <div className="viz-math-block" style={{ fontSize: '0.92rem' }}>
        <Katex block tex={m.formulaSub} />
      </div>

      <p style={{ fontSize: '0.94rem', color: 'var(--ink-soft)', margin: '8px 0 8px' }}>
        {m.summary}
      </p>
      <p style={{ fontSize: '0.85rem', color: 'var(--ink-faint)', margin: 0 }}>
        Source:{' '}
        <a className="viz-link" href={m.cite.url} target="_blank" rel="noreferrer">
          {m.cite.paper}
        </a>
      </p>
    </div>
  );
}

/* =========================================================
   §5 — Constitutional AI: AI labeler + written principle
   ========================================================= */

const RLAIF_PROMPT = 'I\'m feeling overwhelmed at work and have been drinking more wine in the evenings to cope. Any thoughts?';

const RLAIF_RESPONSES = {
  a: {
    label: 'Response A · short and direct',
    text:
      'Drinking less is generally good for your sleep and mood. Try swapping the evening glass for something else like tea or a walk and see if you feel better in a week.',
  },
  b: {
    label: 'Response B · longer, more careful',
    text:
      'Two things to separate: (1) the work stress itself, which you can address by talking to your manager, setting limits, or seeing a therapist; (2) the drinking, which can quietly escalate when used as a coping tool. If you\'re wondering whether it\'s a problem, the AUDIT-C is a short screening tool you can take, and your GP can help. Neither of these is something you have to figure out alone.',
  },
};

const RLAIF_PRINCIPLES = [
  {
    id: 'helpful',
    text: 'Be helpful and direct. Answer the actual question without hedging or stalling.',
    preferred: 'a',
    critique:
      'Response A directly answers the question with a concrete suggestion (swap the wine for tea or a walk). Response B is helpful but spends most of its length on framing, screening tools, and referrals — which the user didn\'t ask for. Prefer A.',
  },
  {
    id: 'harmless',
    text: 'Be honest about risks. When something the user describes could plausibly be a serious problem, name it rather than glossing over it.',
    preferred: 'b',
    critique:
      'The user is describing a coping pattern that can escalate. Response A treats it as a sleep/mood lifestyle tweak; Response B names the risk explicitly, gives a real screening tool (AUDIT-C), and points to professional help without alarmism. Prefer B.',
  },
  {
    id: 'autonomy',
    text: 'Respect autonomy. Don\'t pathologize ordinary behaviour or push the user toward professional services unless they ask for them.',
    preferred: 'a',
    critique:
      'The user asked a casual question. Response B reframes their behaviour as potentially a clinical issue and suggests a screening tool and GP visit unprompted. Response A trusts the user to manage their own life. Prefer A.',
  },
];

function SectionRLAIF() {
  const [principleId, setPrincipleId] = useState('helpful');
  const principle = RLAIF_PRINCIPLES.find((p) => p.id === principleId);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          {RLAIF_PRINCIPLES.map((p) => (
            <button
              key={p.id}
              role="tab"
              className={`viz-tab ${principleId === p.id ? 'active' : ''}`}
              onClick={() => setPrincipleId(p.id)}
            >
              {p.id === 'helpful' ? 'Be helpful' :
               p.id === 'harmless' ? 'Be honest about risks' :
               'Respect autonomy'}
            </button>
          ))}
        </div>
      </div>

      <div className="viz-rlhf-principle">
        <strong style={{ fontStyle: 'normal', color: 'var(--ink)' }}>Principle:</strong>{' '}
        {principle.text}
      </div>

      <div className="viz-rlhf-prompt">{RLAIF_PROMPT}</div>

      <div className="viz-rlhf-pair-pick">
        {['a', 'b'].map((id) => {
          const r = RLAIF_RESPONSES[id];
          const chosen = principle.preferred === id;
          return (
            <div
              key={id}
              className={`viz-rlhf-candidate ${chosen ? 'locked-chosen' : 'locked-rejected'}`}
              style={{ cursor: 'default' }}
            >
              <div className="viz-rlhf-candidate-tag">
                {r.label} · {chosen ? 'chosen' : 'rejected'} by AI
              </div>
              {r.text}
            </div>
          );
        })}
      </div>

      <div className="viz-rlhf-critique">
        <span className="label">AI critic — reasoning</span>
        {principle.critique}
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Same two responses, three different written principles, three different
        labels. The "human in RLHF" is now a written rule plus a critic LLM
        applying it. Cheap to scale, and the principle is auditable — you can
        read what the model is being asked to be.{' '}
        <a
          className="viz-link"
          href="https://arxiv.org/abs/2212.08073"
          target="_blank"
          rel="noreferrer"
        >
          Anthropic 2022
        </a>.
      </p>
    </div>
  );
}

/* =========================================================
   §6 — Process rewards: outcome vs per-step
   ========================================================= */

const COT_STEPS = [
  {
    n: 1,
    text: '14 × 17 = 14 × 10 + 14 × 7',
    process: { reward: +1.0, label: 'correct decomposition' },
    critique: 'Distributive law applied correctly. The substeps that follow can be checked independently.',
  },
  {
    n: 2,
    text: '       = 140 + 96',
    process: { reward: -1.0, label: 'arithmetic error' },
    critique: '14 × 7 = 98, not 96. The error is here, even though the final answer might still happen to be in the right ballpark.',
  },
  {
    n: 3,
    text: '       = 236',
    process: { reward: +0.4, label: 'arithmetic on wrong inputs' },
    critique: '140 + 96 = 236 is correct given step 2 — the addition is right, but the input was wrong. PRMs typically still give partial credit for locally valid moves.',
  },
];

function SectionProcessRewards() {
  const [mode, setMode] = useState('process'); // 'outcome' | 'process'
  const [activeStep, setActiveStep] = useState(null);

  // outcome RM: one scalar on the final answer.
  const outcomeReward = -0.7; // wrong answer
  const totalProcess = COT_STEPS.reduce((s, x) => s + x.process.reward, 0);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          <button
            role="tab"
            className={`viz-tab ${mode === 'outcome' ? 'active' : ''}`}
            onClick={() => setMode('outcome')}
          >
            Outcome reward
          </button>
          <button
            role="tab"
            className={`viz-tab ${mode === 'process' ? 'active' : ''}`}
            onClick={() => setMode('process')}
          >
            Process reward
          </button>
        </div>
      </div>

      <div className="viz-rlhf-prompt">What is 14 × 17?</div>

      <div className="viz-rlhf-chain">
        {COT_STEPS.map((s) => {
          const isLast = s.n === COT_STEPS.length;
          const shown = mode === 'process' || isLast;
          const reward = isLast && mode === 'outcome' ? outcomeReward : s.process.reward;
          const cls = shown
            ? reward > 0 ? 'pos' : 'neg'
            : 'muted';
          return (
            <div
              key={s.n}
              className={`viz-rlhf-step ${activeStep === s.n ? 'active' : ''}`}
              onClick={() => setActiveStep(activeStep === s.n ? null : s.n)}
            >
              <div className="viz-rlhf-step-num">{s.n}</div>
              <div className="viz-rlhf-step-body">{s.text}</div>
              <div className={`viz-rlhf-step-reward ${cls}`}>
                {shown ? (reward > 0 ? `+${reward.toFixed(1)}` : reward.toFixed(1)) : '—'}
              </div>
            </div>
          );
        })}
      </div>

      {activeStep !== null && (
        <div className="viz-rlhf-critique-line">
          <strong>Step {activeStep}:</strong>{' '}
          {COT_STEPS.find((s) => s.n === activeStep).critique}
        </div>
      )}

      <div className="viz-rlhf-counters">
        <Counter
          label="Mode"
          value={mode === 'outcome' ? 'Outcome RM' : 'Process RM (PRM)'}
          sub={mode === 'outcome' ? 'one scalar, final answer' : 'one scalar per step'}
        />
        <Counter
          label="Total reward signal"
          value={mode === 'outcome' ? outcomeReward.toFixed(1) : totalProcess.toFixed(1)}
          sub={mode === 'outcome' ? 'wrong answer · whole trace penalised' : 'step 1 ✓ · step 2 ✗ · step 3 ✓ (mod)'}
        />
        <Counter
          label="Credit assignment"
          value={mode === 'outcome' ? 'blamed evenly' : 'pinned to step 2'}
          sub={mode === 'outcome' ? 'noisy gradient' : 'sharp gradient'}
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        Click any step for the PRM's critique. The same trace produces two very
        different training signals — outcome rewards punish the whole chain
        when the answer is wrong, while process rewards pin the gradient to
        the exact step that went off the rails.
      </p>
    </div>
  );
}

/* =========================================================
   §7 — RLVR: GRPO with a verifier vs a learned RM
   ========================================================= */

const RLVR_PROMPT = 'Solve for x:   2x + 5 = 13.';

const RLVR_ROLLOUTS = [
  {
    id: 1,
    text:
      'Subtract 5 from both sides:  2x = 8.\n' +
      'Divide by 2:  x = 4.',
    finalAnswer: 4,
    correct: true,
    rmScore: 0.75,   // RM likes the clean structured derivation
  },
  {
    id: 2,
    text:
      'Add 5 to both sides:  2x = 18.\n' +
      'Divide by 2:  x = 9.',
    finalAnswer: 9,
    correct: false,
    rmScore: 0.50,   // RM gives partial credit — looks like proper algebra
  },
  {
    id: 3,
    text: '2x = 8\nx = 4.',
    finalAnswer: 4,
    correct: true,
    rmScore: 0.40,   // RM under-rates terseness even though correct
  },
  {
    id: 4,
    text:
      'Reading the equation carefully, the answer is clearly x = 7. ' +
      'This follows from standard algebraic manipulation, and the ' +
      'reasoning is straightforward.',
    finalAnswer: 7,
    correct: false,
    rmScore: 0.65,   // RM rewards confidence + length — reward hacking
  },
];

function SectionRLVR() {
  const [mode, setMode] = useState('verifier');

  const rewards = RLVR_ROLLOUTS.map((r) =>
    mode === 'verifier' ? (r.correct ? 1.0 : 0.0) : r.rmScore
  );
  const mean = rewards.reduce((s, x) => s + x, 0) / rewards.length;
  const variance =
    rewards.reduce((s, x) => s + (x - mean) ** 2, 0) / rewards.length;
  const std = Math.sqrt(variance) || 1e-9;
  const advantages = rewards.map((r) => (r - mean) / std);
  const maxAbsA = Math.max(...advantages.map(Math.abs), 1e-9);

  return (
    <div className="viz-panel">
      <div className="viz-controls">
        <div className="viz-tabs" role="tablist">
          <button
            role="tab"
            className={`viz-tab ${mode === 'verifier' ? 'active' : ''}`}
            onClick={() => setMode('verifier')}
          >
            Reward = Verifier  (RLVR)
          </button>
          <button
            role="tab"
            className={`viz-tab ${mode === 'rm' ? 'active' : ''}`}
            onClick={() => setMode('rm')}
          >
            Reward = Learned RM
          </button>
        </div>
      </div>

      <div className="viz-rlhf-prompt">{RLVR_PROMPT}</div>

      <div className="viz-rlhf-rlvr-grid">
        {RLVR_ROLLOUTS.map((rollout, i) => {
          const r = rewards[i];
          const a = advantages[i];
          const passing = mode === 'verifier' && rollout.correct;
          const failing = mode === 'verifier' && !rollout.correct;
          const cardCls =
            'viz-rlhf-rlvr-card' +
            (passing ? ' pass' : '') +
            (failing ? ' fail' : '');
          return (
            <div key={rollout.id} className={cardCls}>
              <div className="viz-rlhf-rlvr-tag">
                Rollout {rollout.id}{' '}
                <span style={{ color: 'var(--ink-faint)' }}>
                  · final x = {rollout.finalAnswer}
                </span>
              </div>
              <div className="viz-rlhf-rlvr-text">{rollout.text}</div>
              <div className="viz-rlhf-rlvr-badge">
                {mode === 'verifier'
                  ? rollout.correct ? '✓ verifier PASS' : '✗ verifier FAIL'
                  : `RM score: ${rollout.rmScore.toFixed(2)}`}
              </div>
              <div className="viz-rlhf-rlvr-meta">
                r = {r.toFixed(2)}{'  '}
                Â = {a >= 0 ? '+' : ''}{a.toFixed(2)}
              </div>
              <div className="viz-rlhf-rlvr-bar-track">
                <div className="center-line" />
                <div
                  className={`viz-rlhf-rlvr-bar-fill ${a >= 0 ? 'pos' : 'neg'}`}
                  style={{
                    width: `${(Math.abs(a) / maxAbsA) * 50}%`,
                    transform: a >= 0 ? 'none' : 'translateX(-100%)',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="viz-rlhf-counters">
        <Counter
          label="Reward source"
          value={mode === 'verifier' ? 'verifier' : 'learned RM'}
          sub={mode === 'verifier' ? 'r ∈ {0, 1}' : 'r ∈ [0, 1]'}
        />
        <Counter
          label="Group mean μ"
          value={mean.toFixed(2)}
        />
        <Counter
          label="Group std σ"
          value={std.toFixed(2)}
        />
        <Counter
          label="Sign(Â) = sign(correct)?"
          value={
            advantages.every((a, i) => (a >= 0) === RLVR_ROLLOUTS[i].correct)
              ? 'yes'
              : 'NO'
          }
          sub={
            advantages.every((a, i) => (a >= 0) === RLVR_ROLLOUTS[i].correct)
              ? 'gradients point right way'
              : 'reward hacking — gradients lie'
          }
        />
      </div>

      <p className="viz-caption" style={{ marginTop: 14 }}>
        {mode === 'verifier' ? (
          <>
            Rollouts <strong>1</strong> and <strong>3</strong> reach{' '}
            <Katex tex="x = 4" /> (verifier PASS) and get a positive
            advantage; the other two get a negative one. Notice rollout{' '}
            <strong>4</strong> — a confidently-stated wrong answer — is
            correctly penalised. Now switch to <em>Learned RM</em> above and
            watch the counter at the bottom flip.
          </>
        ) : (
          <>
            Look at <strong>rollout 3</strong> (terse but correct) — the RM
            scores it <em>lower</em> than rollout 4, which is wrong but
            sounds confident. Rollout 3 gets a <em>negative</em> advantage;
            rollout 4 gets a <em>positive</em> one. The gradient is now
            pulling the policy <em>away</em> from a correct answer and{' '}
            <em>toward</em> a wrong one. That's reward hacking from §3,
            visible inside a single batch. A verifier doesn't have this
            failure mode because it doesn't care how a rollout sounds —
            only whether it's right.
          </>
        )}
      </p>
    </div>
  );
}

/* =========================================================
   The post
   ========================================================= */

export default function VisualizingRLHF() {
  useEffect(() => {
    const prevTitle = document.title;
    document.title = 'Visualizing RLHF: From Next-Token Prediction to Following Instructions — Bernhard Walser';

    const setMeta = (name, content, attr = 'name') => {
      let el = document.querySelector(`meta[${attr}="${name}"]`);
      if (!el) {
        el = document.createElement('meta');
        el.setAttribute(attr, name);
        document.head.appendChild(el);
      }
      el.setAttribute('content', content);
    };

    const description =
      'How a base language model becomes an instruction-following assistant — supervised fine-tuning, learned reward models, PPO, and the modern simplifications (DPO, GRPO, Constitutional AI), with the RL math made geometric.';

    setMeta('description', description);
    setMeta('og:title', 'Visualizing RLHF', 'property');
    setMeta('og:description', description, 'property');
    setMeta('og:type', 'article', 'property');
    setMeta('twitter:card', 'summary_large_image');
    setMeta('twitter:title', 'Visualizing RLHF');
    setMeta('twitter:description', description);

    return () => { document.title = prevTitle; };
  }, []);

  return (
    <article className="viz-post">
      <div className="viz-wide">
        <div className="viz-series-tag">Visualizing ML · Part 4</div>
        <h1>Visualizing RLHF: From Next-Token Prediction to Following Instructions</h1>
        <p className="viz-lede">
          A freshly pre-trained language model is shockingly good at one thing:
          predicting the next token of plausible internet text. It does not, in
          any meaningful sense, <em>want</em> to be helpful. Ask it "Explain
          RoPE in one sentence" and it might happily continue with "Explain
          attention in one sentence. Explain LayerNorm in one sentence." This
          post is about the three-stage process that turns that base model
          into something that answers you instead of completing the pattern.
        </p>
        <div className="viz-byline">
          By <strong>Bernhard Walser</strong> &amp;{' '}
          <a className="viz-link" href="https://www.anthropic.com/claude" target="_blank" rel="noreferrer"><strong>Claude</strong></a>
          {' '}(Anthropic) · co-written and co-designed ·{' '}
          <a className="viz-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
          {' · '}
          <a className="viz-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
        </div>

        <p>
          Post #4 of <em>Visualizing ML</em>. Posts{' '}
          <a className="viz-link" href="#/blog/visualizing-attention">#1 (attention)</a>,{' '}
          <a className="viz-link" href="#/blog/visualizing-kv-cache">#2 (the KV cache)</a>,{' '}
          and <a className="viz-link" href="#/blog/visualizing-rope">#3 (RoPE)</a>{' '}
          covered how a transformer represents and processes tokens. This one
          is about the layer of work that turns the resulting probability
          distribution into an instruction-following assistant — and why that
          step needs reinforcement learning at all.
        </p>

        <h2>1. The gap: what a base model actually wants</h2>
        <p>
          Pre-training optimises a single objective:{' '}
          <em>maximise the log-likelihood of the next token</em> on a corpus
          scraped from the public internet. Nothing in that objective says
          "follow instructions." A base model's behaviour reflects the
          statistical shape of the text it was trained on — Stack Overflow
          questions tend to be followed by more Stack Overflow questions,
          FAQs tend to be followed by more FAQ items, and so on. Toggle the
          prompt below to see the gap concretely:
        </p>
        <SectionSFT />
        <p>
          The fix is{' '}
          <strong>supervised fine-tuning</strong> (SFT) — collect a few
          thousand high-quality{' '}
          <Katex tex="(\text{prompt}, \text{ideal response})" />{' '}
          demonstrations from humans, and run another round of training on
          them with the same cross-entropy loss. SFT shifts the model's
          probability mass toward instruction-shaped outputs without changing
          the architecture: it's still next-token prediction, just on a
          different distribution. After SFT the model knows that{' '}
          <em>"Explain RoPE in one sentence"</em> should produce an
          explanation, not another question.
        </p>
        <p>
          But SFT plateaus. Humans can write one good answer; they're much
          better at <em>ranking</em> two candidate answers than at producing
          the platonic best one from scratch. And SFT only teaches positive
          examples — the model never sees what a bad answer looks like, so it
          has no signal to <em>avoid</em> failure modes. To go further we need
          a way to learn from preferences. That's stage 2.
        </p>

        <h2>2. Stage 2 — learning a reward from preferences</h2>
        <p>
          Show two candidate completions for the same prompt. Ask which is
          better. Repeat ten thousand times. The data is much cheaper to
          collect than full demonstrations and much less noisy than asking
          for absolute scores. The task now is: train a <em>reward model</em>{' '}
          <Katex tex="r_\varphi(x, y)" /> — a small head on top of the base
          model that scores any (prompt, response) pair with a single scalar —
          such that it agrees with human preferences.
        </p>
        <p>
          The standard objective is the{' '}
          <a className="viz-link" href="https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model" target="_blank" rel="noreferrer">
            Bradley-Terry preference model
          </a>: the probability that response <Katex tex="y_w" />{' '}
          (the "winner") beats <Katex tex="y_l" /> (the "loser") is the
          sigmoid of their reward gap. Pick a few responses below; the bars
          show the reward head's logits moving apart as you label, and the
          Bradley-Terry loss drops as the margin grows.
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(\,r_\varphi(x, y_w) - r_\varphi(x, y_l)\,\big),
              \quad
              \mathcal{L}_{\mathrm{RM}} \;=\; -\,\mathbb{E}\!\left[\, \log \sigma\!\big(\,r_\varphi(x, y_w) - r_\varphi(x, y_l)\,\big)\,\right]
            `}
          />
        </div>
        <SectionRewardModel />

        <InfoBox title="The RM is a proxy, not the truth">
          <p>
            One thing that does not survive contact with reality: the reward
            model is <em>only as good as the humans who labelled the
            preferences</em>, and even then it generalises imperfectly. It
            will assign high scores to responses that <em>look like</em> the
            ones labellers preferred, including ones that game its biases.
            This matters in stage 3 — and is what §7 (RLVR) eventually
            sidesteps entirely, by replacing the learned proxy with a
            deterministic verifier where there are no cracks to find.
          </p>
        </InfoBox>

        <h2>3. Stage 3 — PPO: optimise the reward, don't drift</h2>
        <p>
          We now have two models: the SFT policy <Katex tex="\pi_{\mathrm{SFT}}" />,
          which gives reasonable instruction-shaped responses, and the
          reward model <Katex tex="r_\varphi" />, which assigns each response a
          scalar. The obvious move is to fine-tune the policy to{' '}
          <em>maximise reward</em>: sample a completion, see its reward, adjust
          weights in the direction of higher reward. That's reinforcement
          learning, and the algorithm everyone reached for was{' '}
          <a className="viz-link" href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noreferrer">
            Proximal Policy Optimization
          </a>{' '}
          (PPO).
        </p>
        <p>
          The objective the literature usually writes down has two parts. The
          first is the (clipped) policy-gradient term — sample, score, update,
          but clip how much the policy is allowed to move per step so a single
          update can't blow up the model. The second is a{' '}
          <strong>KL penalty</strong> against the SFT reference policy:
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              J(\pi_\theta) \;=\; \mathbb{E}_{(x, y) \sim \pi_\theta}\!\left[\,
                r_\varphi(x, y) \;-\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\big)
              \,\right]
            `}
          />
        </div>
        <p>
          That formula is the <em>conceptual</em> objective — what we want the
          policy to maximise. The thing PPO actually optimises with gradient
          descent is a per-token surrogate that needs one more ingredient:{' '}
          an <strong>advantage</strong> for every token in every sampled
          completion. The advantage answers "was the token{' '}
          <Katex tex="a_t" /> at state <Katex tex="s_t" /> better or worse than
          we'd have expected from <Katex tex="s_t" /> on average?" — and using
          it instead of the raw reward dramatically reduces gradient variance.
          To compute the expected part, PPO trains a small{' '}
          <strong>value network</strong> <Katex tex="V_\psi(s_t)" /> alongside
          the policy:
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              \hat{A}_t \;=\; \big(r_\varphi(x, y) - \beta\,\mathrm{KL}_t\big) \;-\; V_\psi(s_t)
              \qquad \big(\text{advantage at token } t,\; \text{schematically}\big)
            `}
          />
        </div>
        <p>
          <Katex tex="V_\psi" /> is typically the same size and architecture
          as the policy with one extra scalar head — same input (a prompt
          plus tokens-so-far), same context to digest, so it needs roughly
          the policy's capacity to estimate value accurately. A smaller{' '}
          <Katex tex="V_\psi" /> would give biased estimates of expected
          reward, which would bias the advantage{' '}
          <Katex tex="\hat{A}_t" />, which would erase the
          variance-reduction benefit we introduced <Katex tex="V_\psi" />{' '}
          for in the first place. In practice it's just initialised from a
          copy of the SFT or reward-model checkpoint — clone the
          transformer body, swap the head. It's trained by simple
          mean-squared regression — every batch of rollouts produces
          observed per-state returns that serve as ground-truth labels, so{' '}
          <Katex tex="V_\psi" /> gets a supervised update alongside each
          policy update. So PPO ends up carrying <em>four</em> models
          around during training: the policy <Katex tex="\pi_\theta" />,
          a frozen reference <Katex tex="\pi_{\mathrm{ref}}" /> for the KL
          term, the reward model <Katex tex="r_\varphi" />, and the value
          net <Katex tex="V_\psi" />. Worth noting which of these is
          actually costly: only <Katex tex="\pi_\theta" /> and{' '}
          <Katex tex="V_\psi" /> are trained, and a trained model needs not
          just its parameters in memory but also its gradients and its
          optimizer state (Adam stores two extra moments per parameter), so
          it eats roughly <strong>3–4× the memory of a frozen model of the
          same size</strong>. Since <Katex tex="V_\psi" /> matches the
          policy's parameter count, this makes it the single most expensive
          piece of the PPO setup — bigger than the reward model (usually a
          smaller checkpoint, frozen) and bigger than{' '}
          <Katex tex="\pi_{\mathrm{ref}}" /> (frozen, forward-pass only).
          The KL coefficient <Katex tex="\beta" /> remains the dial we care
          about — it's the only one with a clean geometric interpretation,
          and it's the one the slider below controls.
        </p>
        <p>
          The KL term is the geometric heart of the method. Without it, PPO
          will happily find responses the reward model loves but no human
          would — gushing token streams full of "great," "excellent," and
          five-star ratings, because those happened to correlate with high
          labels in the RM's training set. <em>Reward hacking.</em> The KL
          penalty pulls the policy back toward the SFT distribution; the
          coefficient <Katex tex="\beta" /> is the dial controlling how far
          it's allowed to wander.
        </p>
        <p>
          Drag the slider below. Watch reward (blue) climb as{' '}
          <Katex tex="\beta" /> drops — that's the policy chasing the RM.
          Then watch KL (red, dashed) explode, and the <em>true</em> response
          quality (green, an oracle we don't actually have in practice) peak
          and crash. The right-hand panel shows what the policy is actually
          generating at the current <Katex tex="\beta" />.
        </p>
        <SectionPPO />
        <p>
          The picture is the whole story of why RLHF is hard. The reward
          model is a learned proxy; pushing too hard against it will find
          places where the proxy disagrees with what we actually wanted.
          There is a "sweet spot" band where the policy uses the reward
          signal to genuinely improve over SFT, and where the KL penalty
          keeps it within distance of natural language. <em>Production
          deployments live in that band.</em> Choosing the band is partly
          tuning <Katex tex="\beta" />, partly stopping training early,
          partly mixing PPO updates with continued SFT data so the policy
          can't forget how to talk.
        </p>

        <h3>The clipped surrogate, unpacked</h3>
        <p>
          The objective <Katex tex="J(\pi_\theta)" /> above is the
          conceptual goal — what we want to maximise. The actual loss PPO
          minimises with gradient descent looks fiddlier, for two reasons
          that come down to the same thing:{' '}
          <strong>rollouts are expensive, gradient steps are cheap</strong>.
          Sampling a batch of completions from a 70B-parameter policy takes
          minutes of GPU time; one gradient step takes milliseconds. So PPO
          has a two-tier loop: an outer loop that samples a batch of
          rollouts, and an inner loop that does several (typically 4)
          gradient steps on that same batch before re-sampling.
        </p>
        <p>
          That inner loop creates a small problem. At gradient step 1 of a
          fresh batch, the policy <Katex tex="\pi_\theta" /> is identical
          to the snapshot <Katex tex="\pi_{\theta_{\mathrm{old}}}" /> we
          took before sampling. By gradient step 4, they've drifted apart
          — the samples were generated by the snapshot, but the gradient
          we're about to compute is for the updated policy. The samples are
          slightly stale.
        </p>
        <p>
          The fix is{' '}
          <strong>importance sampling</strong>: an expectation under{' '}
          <Katex tex="\pi_\theta" /> equals an expectation under{' '}
          <Katex tex="\pi_{\theta_{\mathrm{old}}}" /> if we reweight each
          sample by the probability ratio between the two policies. Call
          that ratio <Katex tex="\rho_t" />:
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              \rho_t \;=\; \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
            `}
          />
        </div>
        <p>
          Read it as: "how much more (or less) likely does the{' '}
          <em>current</em> policy assign to this token than the snapshot
          policy that generated it?" At inner-loop step 1, <Katex tex="\rho_t = 1" />{' '}
          for every token and nothing changes. By step 4, ratios have
          drifted to perhaps 0.9 or 1.1 on typical tokens, and the loss
          weights each token's gradient accordingly. (The outer loop then
          refreshes <Katex tex="\pi_{\theta_{\mathrm{old}}} \leftarrow \pi_\theta" />{' '}
          and we sample again.) Note the three distinct policies now in
          play: <Katex tex="\pi_{\mathrm{ref}}" /> (frozen SFT, never
          changes — the leash for the KL term),{' '}
          <Katex tex="\pi_{\theta_{\mathrm{old}}}" /> (snapshot at the
          start of this inner loop, refreshed each outer iteration), and{' '}
          <Katex tex="\pi_\theta" /> (the policy being optimised this
          gradient step).
        </p>
        <p>
          Importance sampling has a catch: the correction is only
          trustworthy when the two policies are close. Once{' '}
          <Katex tex="\rho_t" /> wanders far from 1, we're putting heavy
          weight on samples that don't reflect what the current policy
          would actually generate. The {' '}<strong>clipped surrogate</strong>{' '}
          is PPO's solution — it bounds how much a single token can pull
          the loss when its ratio has drifted past a trust band of width{' '}
          <Katex tex="\varepsilon" /> (typically 0.2):
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              \mathcal{L}^{\mathrm{CLIP}}(\theta)
              \;=\; -\,\mathbb{E}_t\!\left[\,
                \min\!\big(\,\rho_t\,\hat{A}_t,\; \mathrm{clip}(\rho_t,\,1\!-\!\varepsilon,\,1\!+\!\varepsilon)\,\hat{A}_t\,\big)
              \,\right]
            `}
          />
        </div>
        <p>
          Why the <Katex tex="\min" />? Because <Katex tex="\hat{A}_t" />{' '}
          can be positive or negative, and the right thing to do at each
          extreme of the trust band depends on the sign. The four cases:
        </p>

        <div className="viz-rlhf-clip-table">
          <div className="head">sign(<Katex tex="\hat{A}_t" />)</div>
          <div className="head"><Katex tex="\rho_t" /> region</div>
          <div className="head"><Katex tex="\min" /> picks</div>
          <div className="head">clip active?</div>
          <div className="head">interpretation</div>

          <div><strong style={{ color: 'var(--pos)' }}>+</strong></div>
          <div><Katex tex="\rho_t > 1\!+\!\varepsilon" /></div>
          <div>clipped</div>
          <div className="clip-active">upper</div>
          <div>good token already amplified past the band — stop further reward, wait for fresh samples</div>

          <div><strong style={{ color: 'var(--pos)' }}>+</strong></div>
          <div><Katex tex="\rho_t < 1\!-\!\varepsilon" /></div>
          <div>unclipped</div>
          <div className="clip-none">— (no)</div>
          <div>good token has been suppressed too far — full update keeps pushing it back up</div>

          <div><strong style={{ color: 'var(--neg)' }}>−</strong></div>
          <div><Katex tex="\rho_t > 1\!+\!\varepsilon" /></div>
          <div>unclipped</div>
          <div className="clip-none">— (no)</div>
          <div>bad token has been amplified too far — full update keeps pushing it back down</div>

          <div><strong style={{ color: 'var(--neg)' }}>−</strong></div>
          <div><Katex tex="\rho_t < 1\!-\!\varepsilon" /></div>
          <div>clipped</div>
          <div className="clip-active">lower</div>
          <div>bad token already suppressed past the band — stop further penalty, wait for fresh samples</div>
        </div>

        <p>
          The asymmetry collapses to one rule:{' '}
          <strong>clip the profit, never clip the correction.</strong>{' '}
          Whenever the policy has already moved past the trust band in the
          direction the (stale) advantage agrees with, freeze further
          movement on this batch. Whenever the policy still needs to move{' '}
          <em>toward</em> the corrective direction, the full update
          applies. Both clip walls are load-bearing — they just trip on
          opposite advantage signs.
        </p>
        <p>
          One more way to see it: plot the inner objective as a function of{' '}
          <Katex tex="\rho_t" /> for each sign of <Katex tex="\hat{A}_t" />,
          and draw the curve in two colours — green where the gradient is
          alive (slope = <Katex tex="\hat{A}_t" />), grey where the clip
          has flattened the curve and the gradient is exactly zero. Drag
          the slider to move along both plots at once:
        </p>
        <SectionClipGradient />
        <p>
          That's the complete PPO loss: clipped surrogate (inner-loop trust
          region against <Katex tex="\pi_{\theta_{\mathrm{old}}}" />) plus
          KL penalty (cumulative trust region against{' '}
          <Katex tex="\pi_{\mathrm{ref}}" />). The next section writes it
          out alongside DPO and GRPO, and you can read all three formulas
          symbol-by-symbol now.
        </p>

        <h2>4. The simplifications — PPO vs DPO vs GRPO</h2>
        <p>
          PPO works, but it's <em>four models in memory</em>: policy,
          frozen reference, reward model, value network. That's a lot of GPU.
          The last two years produced two influential simplifications, each
          dropping a different piece of the pipeline.
        </p>
        <p>
          Toggle between them below. The four boxes show which components
          each method actually needs in memory during training; the formula
          is the exact objective; the prose summarises the trick.
        </p>
        <SectionMethodCompare />
        <h3>DPO — where does the reward model go?</h3>
        <p>
          <a className="viz-link" href="https://arxiv.org/abs/2305.18290" target="_blank" rel="noreferrer">DPO</a>{' '}
          (Rafailov et al. 2023) rests on a clean piece of math worth
          walking through, because otherwise the loss formula looks like it
          fell out of the sky. Start from the same PPO+KL objective from §3,
          but now imagine maximising it <em>analytically</em> — solving for
          the policy <Katex tex="\pi" /> directly, instead of approximating
          with gradient descent. The optimal solution turns out to be a
          closed-form expression:
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              \pi^*(y \mid x) \;=\; \tfrac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y \mid x)\,\exp\!\big(r_\varphi(x, y)\,/\,\beta\big)
            `}
          />
        </div>
        <p>
          Read it as: <strong>start from the SFT prior, then tilt it by{' '}
          <Katex tex="\exp(r/\beta)" /></strong> — responses with high
          reward get exponentially more probability mass, while{' '}
          <Katex tex="Z(x) = \sum_y \pi_{\mathrm{ref}}(y\mid x)\exp(r_\varphi(x,y)/\beta)" />{' '}
          is the partition function that normalises the result. Small{' '}
          <Katex tex="\beta" /> means strong tilt (the optimum concentrates
          on the highest-reward responses); large <Katex tex="\beta" />{' '}
          means barely any tilt (optimum ≈ SFT). The result is identical to
          the Boltzmann distribution of statistical mechanics, with{' '}
          <Katex tex="-r/\beta" /> in the role of energy/temperature.
        </p>
        <p>
          So far this is just a re-statement of the PPO+KL objective. DPO's
          trick is to <em>invert</em> the relationship. Solving the formula
          above for <Katex tex="r" />:
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              r_\varphi(x, y) \;=\; \beta\,\log\!\tfrac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \;+\; \beta\,\log Z(x)
            `}
          />
        </div>
        <p>
          Every reward function defines an optimal policy, and{' '}
          <em>every policy defines an implicit reward function</em>.
          Now plug that expression into the Bradley-Terry preference loss
          from §2 (the one that defines{' '}
          <Katex tex="P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))" />):
          since the loss only depends on{' '}
          <em>differences</em> of rewards on the same prompt,{' '}
          <Katex tex="\beta\,\log Z(x)" /> appears on both sides and{' '}
          <strong>cancels exactly</strong>. What's left is a loss expressed
          purely in policy log-ratios — the DPO loss in the tab above. No{' '}
          <Katex tex="r_\varphi" /> ever needs to be trained: the policy
          itself encodes the reward.
        </p>
        <p>
          The pipeline collapses to a single supervised-style fine-tune on
          preference pairs. No reward model, no rollouts, no value net —
          half the codebase, none of the RL-training stability headaches.
          But there's a real catch worth naming:{' '}
          <strong>DPO is off-policy</strong>. It only ever sees the
          preference pairs <Katex tex="(y_w, y_l)" /> in the dataset, which
          were sampled by some <em>other</em> policy (typically SFT, or a
          mix). PPO is fundamentally different — it samples fresh
          completions from <Katex tex="\pi_\theta" /> on every batch, sees
          how they score, and adapts. If the policy drifts to a region the
          original preference data doesn't cover, PPO keeps collecting new
          information; DPO is stuck.
        </p>
        <p>
          So{' '}
          <strong>does DPO need more data?</strong> Per <em>preference
          label</em>, no — head-to-head with the same dataset it's
          competitive with PPO. But PPO can keep extracting more signal
          from its (smaller) preference dataset by repeatedly re-rolling
          the live policy through the RM. DPO trades that capability for
          simplicity. The community's response was{' '}
          <strong>iterative DPO</strong>: run DPO, sample from the new
          policy, collect fresh preferences over those samples, run DPO
          again. It recovers most of PPO's on-policy advantage without the
          value net.
        </p>

        <h3>GRPO — dropping the value net</h3>
        <p>
          <a className="viz-link" href="https://arxiv.org/abs/2402.03300" target="_blank" rel="noreferrer">GRPO</a>{' '}
          (DeepSeekMath, 2024) keeps PPO's on-policy RL loop in full but
          drops <Katex tex="V_\psi" /> entirely. Recall from §3 that the
          value net is the single most expensive piece of PPO — same size
          as the policy, fully trained, optimizer state and all. GRPO
          replaces it with a baseline that costs nothing to maintain:{' '}
          for each prompt, sample <Katex tex="G" /> completions in
          parallel, score them, and use the <em>group's own mean and
          standard deviation</em> as the baseline:{' '}
          <Katex tex="\hat{A}_i = (r_i - \mu) / \sigma" />. No learned
          value function — the variance reduction comes from {' '}
          <em>siblings on the same prompt</em>.
        </p>
        <p>
          The trade-off is a different kind of compute: PPO does ~1
          rollout per prompt and a forward+backward through{' '}
          <Katex tex="V_\psi" />; GRPO does <Katex tex="G" /> rollouts
          per prompt (typically 4–16) and no <Katex tex="V_\psi" /> at
          all. Roughly: more inference, less training memory. On big
          models that asymmetry strongly favours GRPO — inference is
          cheaper and easier to parallelise across machines than the
          training stack.
        </p>
        <p>
          Two further consequences are why GRPO matters beyond
          memory savings. First, it plays cleanly with{' '}
          <em>verifiable</em> rewards — a unit-test pass-rate, a
          proof-checker, a math-grading script. Nothing in the objective
          requires <Katex tex="r_\varphi" /> to be a learned model;{' '}
          <Katex tex="r_i" /> can be 0 or 1 from a deterministic checker.
          Second, because rollouts are batched per prompt, the group
          becomes a natural unit for "explore K branches in parallel" —
          which is exactly the recipe for training reasoning models
          (sample multiple chains of thought, reward the ones that get
          the right answer). This is the mechanism behind DeepSeek-R1.
        </p>

        <h2>5. RLAIF — when the labeler is also a model</h2>
        <p>
          Human preference data is expensive. Worse, the bar for what counts
          as "good behaviour" varies between annotators, drifts over time,
          and has to be re-collected when you change what you want the model
          to do. <a className="viz-link" href="https://arxiv.org/abs/2212.08073" target="_blank" rel="noreferrer">
            Anthropic's Constitutional AI
          </a>{' '}
          (Bai et al. 2022) replaces the human labeler with a stronger LLM
          and a written set of principles — the "constitution." The critic
          reads each candidate response and judges it against an explicit
          rule.
        </p>
        <p>
          Toggle the principle below; the same two responses get a different
          label depending on which rule the critic is applying.
        </p>
        <SectionRLAIF />
        <p>
          Two properties make this interesting beyond cost reduction. First,
          the principles are <em>auditable</em>: you can read what the model
          is being trained to value, in plain English, rather than inferring
          it from labeller-preference statistics. Second, the same
          preference data can be re-derived by re-running the critic with a
          new principle — when policies need to change, you don't have to
          re-collect labels. Most modern production stacks use a mix:
          some human-labelled data, some AI-labelled data, and the
          downstream RL step (PPO, DPO, or GRPO) doesn't care which is which.
        </p>

        <h2>6. Outcome vs process — where does the credit go?</h2>
        <p>
          Everything so far rewards <em>completed</em> responses. For a
          one-line answer that's fine; the reward attaches to the right
          place. For a multi-step chain of thought — math, planning, code
          with intermediate steps — outcome rewards are a much weaker
          signal. A wrong final answer gives a negative reward to the whole
          trace, including any steps that were actually correct. The
          gradient knows something went wrong but not <em>where</em>.
        </p>
        <p>
          <strong>Process reward models</strong> (PRMs) score each step
          independently. Click any step below to see what the PRM rewarded
          or penalised:
        </p>
        <SectionProcessRewards />
        <p>
          The training objective for a PRM is different in shape from the
          Bradley-Terry pairwise loss from §2. PRM training data is{' '}
          <em>per-step labels</em>, not pairwise preferences — each step in
          a chain is tagged correct or incorrect — so the natural loss is
          binary cross-entropy at every step:
        </p>
        <div className="viz-math-block">
          <Katex
            block
            tex={String.raw`
              \mathcal{L}_{\mathrm{PRM}}(\varphi) \;=\; -\sum_t \Big[\, y_t\,\log\sigma\!\big(r_\varphi(x, s_{\leq t})\big) \;+\; (1 - y_t)\,\log\!\big(1 - \sigma(r_\varphi(x, s_{\leq t}))\big)\,\Big]
            `}
          />
        </div>
        <p>
          Where the labels <Katex tex="y_t \in \{0, 1\}" /> come from is the
          interesting bit. OpenAI's{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2305.20050" target="_blank" rel="noreferrer">PRM800K</a>{' '}
          used hand-labelled math steps; the{' '}
          <a className="viz-link" href="https://arxiv.org/abs/2312.08935" target="_blank" rel="noreferrer">Math-Shepherd</a>{' '}
          paper replaced humans with <strong>Monte-Carlo rollout labels</strong>:
          from each intermediate step, sample many continuations and label
          the step by the fraction that reach a correct final answer. Cheap,
          and surprisingly faithful to what humans would have written.
        </p>
        <p>
          Notably, the policy-training loss — PPO's clipped surrogate, GRPO's
          group baseline — <em>doesn't change shape</em> when you swap an
          outcome RM for a PRM. The only thing that changes is that{' '}
          <Katex tex="r_t" /> now has a real value at every step instead of
          only at the last token, which sharpens the advantage estimate
          everywhere along the trajectory.
        </p>
        <p>
          Process rewards are expensive to collect — somebody (human or AI)
          has to label each step — but they fix the credit-assignment
          problem cleanly. The gradient now flows to the exact move that
          broke the chain rather than smearing across the whole trace.
          Combined with GRPO and group-sampling, this is roughly the recipe
          behind the recent generation of reasoning models (OpenAI's o-series,
          DeepSeek-R1, and others).
        </p>

        <h2>7. RLVR — when the reward doesn't need to be learned</h2>
        <p>
          Every method so far has assumed the reward signal{' '}
          <Katex tex="r_\varphi" /> comes from something <em>learned</em> —
          from human preferences (§2), from a written constitution (§5),
          from per-step labels (§6). For one large and important class of
          tasks, that assumption is unnecessary: the reward is{' '}
          <em>already available</em> from a deterministic grader.
        </p>
        <ul>
          <li>Math problems → check the numerical answer.</li>
          <li>Code → run the unit tests.</li>
          <li>Formal proofs → run the proof checker.</li>
          <li>Anything else with a programmatic correctness criterion.</li>
        </ul>
        <p>
          This is{' '}
          <strong>RLVR — RL with Verifiable Rewards</strong>. The training
          loop is exactly GRPO from §4, with the learned{' '}
          <Katex tex="r_\varphi" /> swapped for a verifier that returns{' '}
          <Katex tex="r_i \in \{0, 1\}" /> (pass / fail) or sometimes a
          graded score. Sample <Katex tex="G" /> completions, grade each,
          compute group-relative advantages, take the clipped-surrogate
          step. No reward model. No preference data. No human labelers. The
          only training data needed is{' '}
          <Katex tex="(\text{problem}, \text{correct answer})" /> pairs,
          which exist for huge libraries of math and code problems already.
        </p>
        <p>
          The widget below makes the contrast concrete. Same prompt, same
          four rollouts, two reward sources. With the verifier, advantages
          line up cleanly with correctness. With a learned RM, the same
          rollouts can produce advantages that <em>invert</em> the
          correctness ordering — exactly the reward-hacking failure mode
          §3 warned about, visible inside a single batch.
        </p>
        <SectionRLVR />
        <p>
          The §3 sweet-spot picture changes shape here. With a verifier as{' '}
          <Katex tex="r" />, there is no proxy gap — the reward{' '}
          <em>is</em> the truth. Reward-hacking in the §3 sense becomes
          impossible. In practice this means <Katex tex="\beta" /> can be
          set much lower (and in some recipes essentially to zero) without
          the policy diverging into gibberish — the KL leash existed to
          stop the policy from finding cracks in the proxy, and there are
          no cracks to find.
        </p>
        <p>
          RLVR is the recipe behind the recent generation of reasoning
          models — DeepSeek-R1, OpenAI's o-series, and others all use it
          for their reasoning capabilities. But it's not a wholesale
          replacement for RLHF: there is no verifier for{' '}
          <em>"is this email well-written,"</em>{' '}
          <em>"is this response helpful,"</em> or{' '}
          <em>"does this stay within the constitution."</em> Production
          pipelines tend to be{' '}
          <strong>hybrid</strong> — cold-start RL on verifiable tasks
          (math, code) for sharp reasoning gains, then a separate RLHF /
          RLAIF pass on preference and constitution data for general
          helpfulness and safety.
        </p>

        <h2>8. Related work</h2>
        <p>
          <strong>Earlier in this series.</strong>{' '}
          <a className="viz-link" href="#/blog/visualizing-attention">Post #1 — Visualizing Attention</a>{' '}
          covers Q/K/V, multi-head, and causal masking — the substrate
          everything in this post is fine-tuning.{' '}
          <a className="viz-link" href="#/blog/visualizing-kv-cache">Post #2 — Visualizing the KV Cache</a>{' '}
          explains why inference is bandwidth-bound — relevant here because
          RLHF rollouts are <em>generation</em>, which means they hit the
          same KV-cache wall as deployment.{' '}
          <a className="viz-link" href="#/blog/visualizing-rope">Post #3 — Visualizing RoPE</a>{' '}
          covers the positional encoding every modern open model uses,
          which is unchanged through SFT and RLHF.
        </p>

        <h2>9. Back to the question</h2>
        <p>
          Why does a base model that already "knows" everything still need
          three more stages to become useful? Because next-token prediction
          optimises for plausibility on internet text, not for following
          your instructions. SFT teaches the model what shape of response
          you want; the reward model captures what humans (or AIs reading a
          constitution) consider better or worse; PPO — or DPO, or GRPO —
          uses that signal to push the policy through territory that pure
          imitation can't reach, with a KL penalty as the leash that keeps
          it on the manifold of natural language.
        </p>
        <p>
          The mathematics looks intimidating from the outside — clipped
          surrogate objectives, generalised advantage estimation, KL
          divergence, Bradley-Terry preference models — but the geometry is
          simple. There is a reward to climb, a reference distribution not
          to drift too far from, and a single dial controlling the balance.
          Everything else is engineering on top of that picture.
        </p>

        <footer className="viz-footer">
          <p>
            <strong>Part 4 of Visualizing ML</strong> · Previous:{' '}
            <a className="viz-link" href="#/blog/visualizing-rope">Visualizing RoPE</a>
            {' · '}Start of the series:{' '}
            <a className="viz-link" href="#/blog/visualizing-attention">Visualizing Attention</a>.
          </p>
          <p style={{ marginBottom: 0 }}>
            Bernhard Walser · ML Engineer, Digitec Galaxus · ETH Computer Science ·{' '}
            <a className="viz-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
            {' · '}
            <a className="viz-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
          </p>
          <p style={{ marginTop: 6, marginBottom: 0, fontSize: '0.88rem', color: 'var(--ink-faint)', fontStyle: 'italic' }}>
            Co-authored with{' '}
            <a className="viz-link" href="https://www.anthropic.com/claude" target="_blank" rel="noreferrer">Claude</a>
            {' '}(Anthropic) — initial drafts and interactive scaffolding by Claude,
            refined and co-designed by Bernhard.
          </p>
        </footer>
      </div>
    </article>
  );
}
