// src/posts/RlhfAndPpo.jsx
// Aligning LMs · Part 1 of 3 — RLHF and PPO.
// TAGS FOR REGISTRATION: ['rlhf', 'training', 'transformers']
// EXCERPT: How a base language model that just predicts the next token becomes a helpful assistant — SFT, reward modelling, and PPO with its clipped surrogate, walked through with interactive widgets.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import './PostChrome.css';
import './RlhfAndPpo.css';

/* ============================================================
   KaTeX wrapper — uses the CDN-loaded global window.katex
   ============================================================ */
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
      className={`${block ? 'rp-math-block' : 'rp-math-inline'} ${className}`}
    />
  );
}

/* ============================================================
   InfoBox — reusable aside callout
   ============================================================ */
function InfoBox({ title, children }) {
  return (
    <aside className="rp-aside">
      <h4 className="rp-aside-head">
        <span className="rp-aside-tag">i</span>
        {title}
      </h4>
      <div className="rp-aside-body">{children}</div>
    </aside>
  );
}

/* ============================================================
   Counter chip — small KPI strip beneath each interactive
   ============================================================ */
function Counter({ label, value, sub }) {
  return (
    <div className="rp-counter">
      <div className="rp-counter-label">{label}</div>
      <div className="rp-counter-value">{value}</div>
      {sub && <div className="rp-counter-sub">{sub}</div>}
    </div>
  );
}

/* ============================================================
   §1 — SFT: base model vs SFT model, side-by-side
   ============================================================ */
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
    <div className="viz-panel rp-widget">
      <div className="rp-controls">
        <label>
          Prompt
          <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
            {SFT_PROMPTS.map((p) => (
              <option key={p.id} value={p.id}>{p.prompt}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="rp-prompt">{preset.prompt}</div>

      <div className="rp-pair">
        <div className="rp-card">
          <div className="rp-card-tag base">Base LM · next-token prediction</div>
          <div className="rp-completion base">{preset.base}</div>
        </div>
        <div className="rp-card">
          <div className="rp-card-tag sft">SFT · trained on demonstrations</div>
          <div className="rp-completion">{preset.sft}</div>
        </div>
      </div>

      <p className="rp-caption">
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

/* ============================================================
   §2 — Reward model: preference-pair clicker + Bradley-Terry
   ============================================================ */
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
  const [labels, setLabels] = useState({});
  const handleClick = (pairId, choice) => {
    setLabels((prev) => ({ ...prev, [pairId]: choice }));
  };
  const reset = () => setLabels({});

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
      chosenSum -= 0.3;
      rejectedSum += 0.3;
    }
  }

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
    <div className="viz-panel rp-widget">
      <div className="rp-controls">
        <span className="rp-controls-hint">
          Click the response you think is <strong>better</strong> for each prompt:
        </span>
        <button onClick={reset} className="rp-btn">Reset</button>
      </div>

      <div className="rp-pair-stack">
        {RM_PAIRS.map((pair) => {
          const choice = labels[pair.id];
          const aClass = choice
            ? choice === 'a' ? 'rp-candidate locked-chosen' : 'rp-candidate locked-rejected'
            : 'rp-candidate';
          const bClass = choice
            ? choice === 'b' ? 'rp-candidate locked-chosen' : 'rp-candidate locked-rejected'
            : 'rp-candidate';
          return (
            <div key={pair.id}>
              <div className="rp-prompt">{pair.prompt}</div>
              <div className="rp-pair-pick">
                <button
                  type="button"
                  className={aClass}
                  onClick={() => !choice && handleClick(pair.id, 'a')}
                  disabled={!!choice}
                >
                  <div className="rp-candidate-tag">
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
                  <div className="rp-candidate-tag">
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

      <div className="rp-reward-head">
        <h4 className="rp-reward-head-title">Reward head — Σ over labelled pairs</h4>
        <div className="rp-reward-bars">
          <div>r<sub>φ</sub>(chosen)</div>
          <div className="rp-reward-track">
            <div className="center-line" />
            <div
              className={`rp-reward-fill ${chosenSum >= 0 ? 'pos' : 'neg'}`}
              style={{
                width: `${barWidth(chosenSum) / 2}%`,
                transform: chosenSum >= 0 ? 'none' : 'translateX(-100%)',
              }}
            />
          </div>
          <div className="rp-reward-value">{chosenSum.toFixed(2)}</div>

          <div>r<sub>φ</sub>(rejected)</div>
          <div className="rp-reward-track">
            <div className="center-line" />
            <div
              className={`rp-reward-fill ${rejectedSum >= 0 ? 'pos' : 'neg'}`}
              style={{
                width: `${barWidth(rejectedSum) / 2}%`,
                transform: rejectedSum >= 0 ? 'none' : 'translateX(-100%)',
              }}
            />
          </div>
          <div className="rp-reward-value">{rejectedSum.toFixed(2)}</div>
        </div>
      </div>

      <div className="rp-counters">
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

/* ============================================================
   §3 — PPO: KL coefficient slider + reward/KL plot + sample
   ============================================================ */
function ppoCurves(beta) {
  const t = 1 / (1 + 8 * beta);
  const reward = 0.30 + 0.70 * Math.pow(t, 0.55);
  const kl = 8.0 * Math.pow(t, 2.4);
  const trueQ = 0.32 + 0.55 * Math.pow(t, 0.55)
              - 0.95 * Math.pow(Math.max(0, t - 0.55), 2.2);
  return { t, reward, kl, trueQ };
}

const PPO_SAMPLES = [
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
  const [sliderPos, setSliderPos] = useState(0.55);
  const beta = 0.005 * Math.pow(1000, sliderPos);
  const cur = ppoCurves(beta);

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

  const W = 520, H = 240;
  const padL = 38, padR = 38, padT = 14, padB = 26;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const klMax = 8;
  const yReward = (r) => padT + (1 - r) * innerH;
  const yKL = (k) => padT + (1 - Math.min(1, k / klMax)) * innerH;
  const xPos = (x) => padL + x * innerW;

  const rewardPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${xPos(p.x).toFixed(2)} ${yReward(p.reward).toFixed(2)}`).join(' ');
  const klPath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${xPos(p.x).toFixed(2)} ${yKL(p.kl).toFixed(2)}`).join(' ');
  const truePath = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${xPos(p.x).toFixed(2)} ${yReward(p.trueQ).toFixed(2)}`).join(' ');

  const tToX = (t) => {
    const b = (1 / t - 1) / 8;
    const x = Math.log(b / 0.005) / Math.log(1000);
    return Math.max(0, Math.min(1, x));
  };
  const xFrozenEnd = tToX(0.20);
  const xSweetEnd = tToX(0.80);

  return (
    <div className="viz-panel rp-widget">
      <div className="rp-controls">
        <label className="rp-slider-label">
          KL coefficient&nbsp;<Katex tex="\beta" /> ={' '}
          <strong className="rp-slider-val">
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
          className="rp-range"
        />
      </div>

      <div className="rp-ppo-loop">
        <div>
          <div className="rp-plot-tag">
            reward, KL, and true quality vs <em>β</em> (log scale)
          </div>
          <svg className="rp-ppo-plot" viewBox={`0 0 ${W} ${H}`} width="100%">
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

            <line
              x1={xPos(sliderPos)}
              y1={padT}
              x2={xPos(sliderPos)}
              y2={padT + innerH}
              stroke="var(--post-accent)"
              strokeWidth="1.5"
            />

            <path d={rewardPath} stroke="hsl(218, 60%, 45%)" strokeWidth="2.2" fill="none" />
            <path d={klPath} stroke="hsl(8, 60%, 50%)" strokeWidth="2.2" fill="none" strokeDasharray="5 4" />
            <path d={truePath} stroke="hsl(148, 45%, 38%)" strokeWidth="2.2" fill="none" />

            <circle cx={xPos(sliderPos)} cy={yReward(cur.reward)} r="4" fill="hsl(218, 60%, 45%)" />
            <circle cx={xPos(sliderPos)} cy={yKL(cur.kl)} r="4" fill="hsl(8, 60%, 50%)" />
            <circle cx={xPos(sliderPos)} cy={yReward(cur.trueQ)} r="4" fill="hsl(148, 45%, 38%)" />

            <text x={padL} y={H - 8} fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">β = 0.005</text>
            <text x={padL + innerW} y={H - 8} textAnchor="end" fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">β = 5</text>
            <text x={padL + innerW / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">
              ← unconstrained reward&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;frozen at SFT →
            </text>

            <text x={padL - 4} y={padT + 6} textAnchor="end" fontSize="10" fill="hsl(218, 60%, 45%)" fontFamily="var(--post-mono)">reward 1.0</text>
            <text x={padL - 4} y={padT + innerH - 2} textAnchor="end" fontSize="10" fill="hsl(218, 60%, 45%)" fontFamily="var(--post-mono)">0</text>
            <text x={padL + innerW + 4} y={padT + 6} textAnchor="start" fontSize="10" fill="hsl(8, 60%, 50%)" fontFamily="var(--post-mono)">KL {klMax}</text>
            <text x={padL + innerW + 4} y={padT + innerH - 2} textAnchor="start" fontSize="10" fill="hsl(8, 60%, 50%)" fontFamily="var(--post-mono)">0</text>
          </svg>

          <div className="rp-legend">
            <span><span className="rp-legend-swatch" style={{ background: 'hsl(218, 60%, 45%)' }} />reward r<sub>φ</sub></span>
            <span><span className="rp-legend-swatch" style={{ background: 'hsl(8, 60%, 50%)' }} />KL(π<sub>θ</sub> ‖ π<sub>ref</sub>)</span>
            <span><span className="rp-legend-swatch" style={{ background: 'hsl(148, 45%, 38%)' }} />true quality</span>
          </div>
        </div>

        <div className="rp-ppo-sample">
          <div className="rp-ppo-sample-tag">
            policy output · prompt: "Explain RoPE in one sentence."
          </div>
          <div>
            <span className={`rp-ppo-zone ${sample.zone}`}>{sample.label}</span>
          </div>
          <div className="rp-ppo-sample-body">{sample.text}</div>
        </div>
      </div>

      <div className="rp-counters">
        <Counter label="Reward r_φ" value={cur.reward.toFixed(3)} sub="what PPO optimises" />
        <Counter label="KL from π_ref" value={cur.kl.toFixed(2)} sub="nats" />
        <Counter
          label="True quality"
          value={cur.trueQ.toFixed(3)}
          sub={cur.trueQ < 0.45 && cur.t > 0.55 ? 'RM is being gamed' : 'oracle (not observed)'}
        />
      </div>
    </div>
  );
}

/* ============================================================
   §4 — Clipped-surrogate gradient visualisation
   ============================================================ */
function ClipPlot({ advantage, rho, epsilon, label }) {
  const W = 320, H = 200;
  const padL = 38, padR = 14, padT = 14, padB = 30;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const rhoMin = 0.4, rhoMax = 1.8;
  const yMin = -1.8, yMax = 1.8;

  const toX = (r) => padL + ((r - rhoMin) / (rhoMax - rhoMin)) * innerW;
  const toY = (y) => padT + ((yMax - y) / (yMax - yMin)) * innerH;

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

  const frozenLeft = advantage < 0;
  const threshold = frozenLeft ? 1 - epsilon : 1 + epsilon;
  const activePath = frozenLeft ? buildPath(threshold, rhoMax) : buildPath(rhoMin, threshold);
  const frozenPath = frozenLeft ? buildPath(rhoMin, threshold) : buildPath(threshold, rhoMax);

  const unclippedRefPath = `M ${toX(rhoMin)} ${toY(rhoMin * advantage)} L ${toX(rhoMax)} ${toY(rhoMax * advantage)}`;
  const clippedRefPath = (() => {
    const y0 = (1 - epsilon) * advantage;
    const y1 = (1 + epsilon) * advantage;
    return `M ${toX(rhoMin)} ${toY(y0)} L ${toX(1 - epsilon)} ${toY(y0)}` +
           ` L ${toX(1 + epsilon)} ${toY(y1)} L ${toX(rhoMax)} ${toY(y1)}`;
  })();

  const yCur = Math.min(rho * advantage, Math.max(1 - epsilon, Math.min(1 + epsilon, rho)) * advantage);
  const isFrozen = frozenLeft ? rho < 1 - epsilon : rho > 1 + epsilon;
  const gradient = isFrozen ? 0 : advantage;

  const ACTIVE = 'hsl(148, 45%, 38%)';
  const FROZEN = 'hsl(218, 20%, 60%)';

  return (
    <div>
      <div className="rp-clip-label">{label}</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="rp-clip-svg">
        <rect
          x={toX(1 - epsilon)}
          y={padT}
          width={toX(1 + epsilon) - toX(1 - epsilon)}
          height={innerH}
          fill="hsl(218, 30%, 94%)"
        />
        <line x1={padL} y1={toY(0)} x2={padL + innerW} y2={toY(0)} stroke="hsl(218, 20%, 80%)" />
        <line x1={toX(1)} y1={padT} x2={toX(1)} y2={padT + innerH} stroke="hsl(218, 20%, 80%)" strokeDasharray="2 3" />
        <line x1={toX(1 - epsilon)} y1={padT} x2={toX(1 - epsilon)} y2={padT + innerH}
              stroke="hsl(218, 30%, 70%)" strokeDasharray="3 3" strokeWidth="1" />
        <line x1={toX(1 + epsilon)} y1={padT} x2={toX(1 + epsilon)} y2={padT + innerH}
              stroke="hsl(218, 30%, 70%)" strokeDasharray="3 3" strokeWidth="1" />

        <path d={unclippedRefPath} stroke="hsl(218, 30%, 75%)" strokeWidth="1" fill="none" strokeDasharray="3 3" />
        <path d={clippedRefPath} stroke="hsl(218, 30%, 75%)" strokeWidth="1" fill="none" strokeDasharray="3 3" />

        <path d={frozenPath} stroke={FROZEN} strokeWidth="3" fill="none" />
        <path d={activePath} stroke={ACTIVE} strokeWidth="3" fill="none" />

        <line x1={toX(rho)} y1={padT} x2={toX(rho)} y2={padT + innerH}
              stroke="var(--post-accent)" strokeWidth="1.5" />
        <circle cx={toX(rho)} cy={toY(yCur)} r="5"
                fill={isFrozen ? FROZEN : ACTIVE} stroke="#fff" strokeWidth="1.5" />

        <text x={padL} y={H - 12} fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">ρ = {rhoMin}</text>
        <text x={toX(1)} y={H - 12} textAnchor="middle" fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">1</text>
        <text x={toX(1 - epsilon)} y={H - 2} textAnchor="middle" fontSize="9" fill="hsl(218, 30%, 50%)" fontFamily="var(--post-mono)">1−ε</text>
        <text x={toX(1 + epsilon)} y={H - 2} textAnchor="middle" fontSize="9" fill="hsl(218, 30%, 50%)" fontFamily="var(--post-mono)">1+ε</text>
        <text x={padL + innerW} y={H - 12} textAnchor="end" fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">{rhoMax}</text>

        <text x={padL - 4} y={toY(advantage > 0 ? 1.5 : -1.5) + 3} textAnchor="end" fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">
          {advantage > 0 ? '+1.5' : '−1.5'}
        </text>
        <text x={padL - 4} y={toY(0) + 3} textAnchor="end" fontSize="10" fill="var(--post-text-mut)" fontFamily="var(--post-mono)">0</text>
      </svg>

      <div className="rp-clip-legend">
        <span className={`rp-grad-chip ${isFrozen ? 'frozen' : 'active'}`}>
          ∂L/∂ρ = {gradient === 0 ? '0  (frozen)' : (gradient > 0 ? '+1' : '−1')}
        </span>
        <span className="rp-grad-rho">point at ρ = {rho.toFixed(2)}</span>
      </div>
    </div>
  );
}

function SectionClipGradient() {
  const [rho, setRho] = useState(1.0);
  const epsilon = 0.2;
  return (
    <div className="viz-panel rp-widget">
      <div className="rp-controls">
        <label className="rp-slider-label">
          importance ratio&nbsp;<Katex tex="\rho_t" /> ={' '}
          <strong className="rp-slider-val">{rho.toFixed(2)}</strong>
        </label>
        <input
          type="range"
          min="0.4"
          max="1.8"
          step="0.01"
          value={rho}
          onChange={(e) => setRho(parseFloat(e.target.value))}
          className="rp-range"
        />
      </div>

      <div className="rp-clip-plots">
        <ClipPlot advantage={+1} rho={rho} epsilon={epsilon} label="Â_t = +1   (good action)" />
        <ClipPlot advantage={-1} rho={rho} epsilon={epsilon} label="Â_t = −1   (bad action)" />
      </div>

      <p className="rp-caption">
        Solid green = gradient is alive (<Katex tex="\partial L / \partial \rho = \hat{A}_t" />).
        Solid grey = gradient is zero (<Katex tex="\partial L / \partial \rho = 0" />,
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

/* ============================================================
   Page
   ============================================================ */
export default function RlhfAndPpo() {
  useEffect(() => {
    const prev = document.title;
    document.title = 'RLHF and PPO — Bernhard Walser';
    return () => { document.title = prev; };
  }, []);

  return (
    <article className="post-2026 rp-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />
            Aligning LMs · Part 1 of 3
          </div>
          <h1>RLHF and PPO</h1>
          <p className="post-lede">
            A base language model can finish your sentence, but it's not yet an
            assistant — it just predicts the next token. <em>Reinforcement
            learning from human feedback</em> is the layer that turns that
            probability machine into something helpful, honest, and safe. Three
            stages: supervised fine-tuning, a learned reward, and PPO.
          </p>
          <div className="post-byline">
            By <strong>Bernhard Walser</strong> &amp;{' '}
            <a className="post-link" href="https://www.anthropic.com/claude" target="_blank" rel="noreferrer"><strong>Claude</strong></a>
            {' '}(Anthropic) · co-written and co-designed ·{' '}
            <a className="post-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
            {' · '}
            <a className="post-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
          </div>
        </div>

        <p>
          The recipe was popularised by{' '}
          <a className="post-link" href="https://arxiv.org/abs/1706.03741" target="_blank" rel="noreferrer">Christiano et al. 2017</a>{' '}
          and scaled to language models in{' '}
          <a className="post-link" href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noreferrer">InstructGPT (Ouyang et al. 2022)</a>.
          The pipeline below is the one every aligned model since — GPT-4,
          Claude, Gemini — was first built on.
        </p>

        <h2 className="reveal">1. The gap: what a base model actually wants</h2>
        <p>
          Pre-training optimises a single objective:{' '}
          <em>maximise the log-likelihood of the next token</em> on a corpus
          scraped from the public internet. Nothing in that objective says
          "follow instructions." A base model's behaviour reflects the
          statistical shape of the text it was trained on — Stack Overflow
          questions tend to be followed by more Stack Overflow questions,
          FAQs tend to be followed by more FAQ items, and so on. Toggle the
          prompt below to see the gap concretely.
        </p>
        <SectionSFT />
        <p>
          The fix is <strong>supervised fine-tuning</strong> (SFT) — collect a
          few thousand high-quality{' '}
          <Katex tex="(\text{prompt}, \text{ideal response})" /> demonstrations
          from humans, and run another round of training on them with the same
          cross-entropy loss. SFT shifts the model's probability mass toward
          instruction-shaped outputs without changing the architecture: it's
          still next-token prediction, just on a different distribution.
        </p>
        <p>
          But SFT plateaus. Humans can write one good answer; they're much
          better at <em>ranking</em> two candidate answers than at producing
          the platonic best one from scratch. And SFT only teaches positive
          examples — the model never sees what a bad answer looks like, so it
          has no signal to <em>avoid</em> failure modes. To go further we need
          a way to learn from preferences. That's stage 2.
        </p>

        <h2 className="reveal">2. Stage 2 — learning a reward from preferences</h2>
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
          <a className="post-link" href="https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model" target="_blank" rel="noreferrer">
            Bradley-Terry preference model
          </a>: the probability that response <Katex tex="y_w" /> (the
          "winner") beats <Katex tex="y_l" /> (the "loser") is the sigmoid of
          their reward gap. Pick a few responses below; the bars show the
          reward head's logits moving apart as you label, and the
          Bradley-Terry loss drops as the margin grows.
        </p>
        <div className="rp-math-block">
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
            That matters in stage 3.
          </p>
        </InfoBox>

        <h2 className="reveal">3. Stage 3 — PPO: optimise the reward, don't drift</h2>
        <p>
          We now have two models: the SFT policy <Katex tex="\pi_{\mathrm{SFT}}" />,
          which gives reasonable instruction-shaped responses, and the reward
          model <Katex tex="r_\varphi" />, which assigns each response a scalar.
          The obvious move is to fine-tune the policy to{' '}
          <em>maximise reward</em>: sample a completion, see its reward, adjust
          weights in the direction of higher reward. That's reinforcement
          learning, and the algorithm everyone reached for was{' '}
          <a className="post-link" href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noreferrer">
            Proximal Policy Optimization
          </a> (PPO, Schulman et al. 2017).
        </p>
        <p>
          The objective has two parts. The first is a (clipped) policy-gradient
          term — sample, score, update, but clip how much the policy is allowed
          to move per step so a single update can't blow up the model. The
          second is a <strong>KL penalty</strong> against the SFT reference
          policy:
        </p>
        <div className="rp-math-block">
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
          That formula is the <em>conceptual</em> objective. What PPO actually
          optimises with gradient descent is a per-token surrogate that needs
          one more ingredient: an <strong>advantage</strong>{' '}
          <Katex tex="\hat{A}_t" /> for every token in every sampled
          completion. The advantage answers "was this token better or worse
          than we'd have expected from the current state on average?" — and
          using it instead of the raw reward dramatically reduces gradient
          variance. To estimate the expected part, PPO trains a small{' '}
          <strong>value network</strong> <Katex tex="V_\psi(s_t)" /> alongside
          the policy, typically initialised from a copy of the SFT or
          reward-model checkpoint and trained by mean-squared regression
          against observed returns.
        </p>
        <p>
          So PPO ends up carrying <em>four</em> models at training time: the
          policy <Katex tex="\pi_\theta" />, the frozen reference{' '}
          <Katex tex="\pi_{\mathrm{ref}}" />, the reward model{' '}
          <Katex tex="r_\varphi" />, and the value net{' '}
          <Katex tex="V_\psi" />. The value net is the single most
          expensive piece — the same size as the policy, fully trained,
          optimizer state and all. The KL coefficient{' '}
          <Katex tex="\beta" /> is the dial we care about most: it has a clean
          geometric interpretation, and it's the one the slider below
          controls.
        </p>
        <p>
          The KL term is the geometric heart of the method. Without it, PPO
          will happily find responses the reward model loves but no human
          would — gushing token streams full of "great," "excellent," and
          five-star ratings, because those happened to correlate with high
          labels in the RM's training set. <em>Reward hacking.</em> The KL
          penalty pulls the policy back toward the SFT distribution.
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
          That's the whole story of why RLHF is hard. The reward model is a
          learned proxy; push too hard and you find places where the proxy
          disagrees with what you actually wanted. There's a sweet-spot band
          where the policy uses the reward signal to genuinely improve over
          SFT, and where the KL penalty keeps it within distance of natural
          language. <em>Production deployments live in that band.</em>
        </p>

        <h2 className="reveal">4. The clipped surrogate, unpacked</h2>
        <p>
          The objective <Katex tex="J(\pi_\theta)" /> above is the
          conceptual goal. The actual loss PPO minimises looks fiddlier, for
          one practical reason:{' '}
          <strong>rollouts are expensive, gradient steps are cheap</strong>.
          Sampling a batch of completions from a 70B-parameter policy takes
          minutes of GPU time; one gradient step takes milliseconds. So PPO
          uses a two-tier loop — an outer loop that samples a batch of
          rollouts, and an inner loop that does several (typically 4)
          gradient steps on the same batch before re-sampling.
        </p>
        <p>
          That inner loop creates a problem. By step 4 the policy{' '}
          <Katex tex="\pi_\theta" /> has drifted from the snapshot{' '}
          <Katex tex="\pi_{\theta_{\mathrm{old}}}" /> that generated the
          samples. The fix is <strong>importance sampling</strong>: reweight
          each sample by the probability ratio between the two policies.
          Call that ratio <Katex tex="\rho_t" />:
        </p>
        <div className="rp-math-block">
          <Katex
            block
            tex={String.raw`
              \rho_t \;=\; \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
            `}
          />
        </div>
        <p>
          But the correction is only trustworthy when the two policies are
          close. Once <Katex tex="\rho_t" /> wanders far from 1, we're putting
          heavy weight on samples that don't reflect what the current policy
          would actually generate. The <strong>clipped surrogate</strong>{' '}
          is PPO's solution — bound how much a single token can pull the loss
          when its ratio has drifted past a trust band of width{' '}
          <Katex tex="\varepsilon" /> (typically 0.2):
        </p>
        <div className="rp-math-block">
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
          The <Katex tex="\min" /> is there because <Katex tex="\hat{A}_t" />{' '}
          can be positive or negative, and the right thing to do at each
          extreme of the trust band depends on the sign. The asymmetry
          collapses to one rule:{' '}
          <strong>clip the profit, never clip the correction.</strong>{' '}
          Whenever the policy has already moved past the trust band in the
          direction the (stale) advantage agrees with, freeze further
          movement. Whenever the policy still needs to move{' '}
          <em>toward</em> the corrective direction, the full update
          applies.
        </p>
        <p>
          Plot the inner objective as a function of <Katex tex="\rho_t" /> for
          each sign of <Katex tex="\hat{A}_t" />, and draw the curve in two
          colours — green where the gradient is alive (slope ={' '}
          <Katex tex="\hat{A}_t" />), grey where the clip has flattened the
          curve and the gradient is exactly zero. Drag the slider:
        </p>
        <SectionClipGradient />

        <h2 className="reveal">Four networks is a lot</h2>
        <p>
          That's the canonical pipeline: SFT, then a reward model, then PPO
          with its clipped surrogate and its KL leash. It works — InstructGPT,
          GPT-4, Claude, Gemini are all built on this foundation. But it
          carries <em>four</em> networks in GPU memory at training time, and
          it's famously fiddly to tune.
        </p>
        <p>
          The last two years of alignment research are mostly the same story
          told in different ways: do more with less. Drop the reward model.
          Drop the value net. Patch the failure modes that show up at scale.
          That's <a className="post-link" href="#/blog/beyond-ppo">Part 2</a>.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Christiano et al. 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1706.03741" target="_blank" rel="noreferrer">Deep Reinforcement Learning from Human Preferences</a>
            </div>
            <div className="ref-note">The paper that introduced the RLHF recipe — preference labels, learned reward, RL.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03741" target="_blank" rel="noreferrer">arxiv.org/abs/1706.03741</a></div>

          <div className="ref-cite">Ouyang et al. 2022</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noreferrer">Training language models to follow instructions with human feedback (InstructGPT)</a>
            </div>
            <div className="ref-note">RLHF applied to language models at scale — the three-stage pipeline this post walks through.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/2203.02155" target="_blank" rel="noreferrer">arxiv.org/abs/2203.02155</a></div>

          <div className="ref-cite">Schulman et al. 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noreferrer">Proximal Policy Optimization Algorithms</a>
            </div>
            <div className="ref-note">PPO — the clipped surrogate and trust-region machinery used in stage 3.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1707.06347" target="_blank" rel="noreferrer">arxiv.org/abs/1707.06347</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 1 of Aligning LMs</strong> · Next:{' '}
            <a className="post-link" href="#/blog/beyond-ppo">Beyond PPO (DPO, GRPO, DAPO)</a>
            {' · '}Continues with:{' '}
            <a className="post-link" href="#/blog/beyond-human-feedback">Beyond Human Feedback (RLAIF, Process Rewards, RLVR)</a>.
          </p>
          <p style={{ marginBottom: 0 }}>
            Bernhard Walser · ML Engineer, Digitec Galaxus · ETH Computer Science ·{' '}
            <a className="post-link" href="https://www.linkedin.com/in/bernhardwalser/" target="_blank" rel="noreferrer">LinkedIn</a>
            {' · '}
            <a className="post-link" href="https://github.com/berniwal" target="_blank" rel="noreferrer">GitHub</a>
          </p>
        </footer>
      </div>
    </article>
  );
}
