// TAGS FOR REGISTRATION: ['nlp', 'history', 'transformers']
// EXCERPT: How NLP got from counting words to attention — tokens, TF-IDF, embeddings, and the RNN/CNN limits that attention was built to escape. Part 1 of a 3-part series on Transformers.
//
// Transformers · Part 1 of 3 — From TF-IDF to Attention.
// Ported from the original Visualizing Attention markdown post; widgets reused
// from VisualizingAttention.jsx, restyled for the post-2026 chrome.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import usePageMeta from '../usePageMeta';
import './PostChrome.css';
import './NlpHistory.css';

/* ============================================================
   KaTeX wrapper — uses the global window.katex loaded in
   public/index.html.
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
      className={`${block ? 'nlp-math-block' : 'nlp-math-inline'} ${className}`}
    />
  );
}

/* ============================================================
   Shared data
   ============================================================ */
const fakeTokenId = (s) => {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) >>> 0;
  }
  return h % 49997;
};

const TOKENIZER_PRESETS = [
  {
    id: 'aiayn',
    text: 'Attention is all you need',
    splits: {
      char: 'Attention is all you need'.split(''),
      word: ['Attention', 'is', 'all', 'you', 'need'],
      subword: ['Att', 'ention', ' is', ' all', ' you', ' need'],
    },
  },
  {
    id: 'unhappy',
    text: 'unhappiness',
    splits: {
      char: 'unhappiness'.split(''),
      word: ['unhappiness'],
      subword: ['un', 'happi', 'ness'],
    },
  },
  {
    id: 'trans',
    text: 'transformer-based',
    splits: {
      char: 'transformer-based'.split(''),
      word: ['transformer-based'],
      subword: ['transform', 'er', '-', 'based'],
    },
  },
  {
    id: 'gpt',
    text: 'GPT-4 costs $0.03',
    splits: {
      char: 'GPT-4 costs $0.03'.split(''),
      word: ['GPT-4', 'costs', '$0.03'],
      subword: ['GPT', '-', '4', ' costs', ' $', '0', '.', '03'],
    },
  },
];

// Small hand-rounded 8-dim embedding table, reused from the original post.
const EMB = {
  the:    [ 0.2, -0.1,  0.3,  0.0,  0.1, -0.2,  0.4,  0.1],
  cat:    [ 0.5,  0.3, -0.2,  0.4,  0.1,  0.0, -0.1,  0.2],
  sat:    [-0.1,  0.4,  0.5,  0.2, -0.3,  0.1,  0.2, -0.4],
  on:     [ 0.1, -0.2,  0.1, -0.3,  0.4,  0.2, -0.1,  0.0],
  mat:    [ 0.4,  0.2, -0.1,  0.3, -0.2,  0.5,  0.0,  0.1],
  she:    [ 0.6,  0.1,  0.0,  0.2,  0.3, -0.1,  0.4, -0.2],
  gave:   [-0.2,  0.5,  0.3, -0.1,  0.1,  0.4, -0.3,  0.2],
  him:    [ 0.4, -0.1,  0.2,  0.5, -0.2,  0.0,  0.3,  0.1],
  a:      [ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
  book:   [-0.3,  0.2,  0.5, -0.1,  0.4,  0.2, -0.2,  0.3],
  quick:  [ 0.3,  0.4, -0.2,  0.1,  0.5, -0.1,  0.2,  0.0],
  brown:  [ 0.2, -0.3,  0.1,  0.4, -0.1,  0.3,  0.0,  0.2],
  fox:    [-0.1,  0.5,  0.2, -0.2,  0.3,  0.1,  0.4, -0.3],
  jumps:  [ 0.0,  0.2, -0.3,  0.5,  0.1, -0.2,  0.3,  0.4],
  code:   [ 0.5, -0.2,  0.4,  0.1, -0.3,  0.2,  0.1,  0.0],
  that:   [ 0.1,  0.3, -0.1,  0.0,  0.2, -0.2,  0.4,  0.1],
  writes: [-0.2,  0.4,  0.3,  0.5,  0.0,  0.1, -0.3,  0.2],
  itself: [ 0.3, -0.1,  0.5,  0.2, -0.2,  0.4,  0.0, -0.1],
};

const SENTENCES = [
  { id: 'cat',   tokens: ['the', 'cat', 'sat', 'on', 'the', 'mat'] },
  { id: 'she',   tokens: ['she', 'gave', 'him', 'a', 'book'] },
  { id: 'fox',   tokens: ['a', 'quick', 'brown', 'fox', 'jumps'] },
  { id: 'code',  tokens: ['code', 'that', 'writes', 'itself'] },
];

/* ============================================================
   Widget 1 — Tokenization (char / word / subword)
   ============================================================ */
function SectionTokens() {
  const [presetId, setPresetId] = useState('aiayn');
  const [mode, setMode] = useState('subword');
  const preset = TOKENIZER_PRESETS.find((p) => p.id === presetId);
  const tokens = preset.splits[mode];

  return (
    <div className="viz-panel">
      <div className="nlp-controls">
        <label>
          Input
          <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
            {TOKENIZER_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.text}</option>
            ))}
          </select>
        </label>
        <div className="nlp-tabs" role="tablist">
          {['char', 'word', 'subword'].map((m) => (
            <button
              key={m}
              role="tab"
              className={`nlp-tab ${mode === m ? 'active' : ''}`}
              onClick={() => setMode(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="nlp-token-source">"{preset.text}"</div>

      <div className="nlp-token-row">
        {tokens.map((t, i) => (
          <div className="nlp-token-pill" key={i}>
            <div className="nlp-token-text">{t === ' ' ? '␣' : t.replace(/^ /, '·')}</div>
            <div className="nlp-token-id">{fakeTokenId(t)}</div>
          </div>
        ))}
      </div>

      <p className="viz-caption">
        <strong>{tokens.length} token{tokens.length === 1 ? '' : 's'}.</strong>{' '}
        Each pill is one entry in the model's vocabulary. The integer below it is
        the token ID — that's what actually flows into the network.
        Leading-space markers (<code>·</code>) are how real BPE tokenizers
        preserve word boundaries.
      </p>
    </div>
  );
}

/* ============================================================
   Widget 2 — TF-IDF heatmap on a tiny corpus
   ============================================================ */
const TFIDF_CORPUS = [
  ['the', 'cat', 'sat', 'on', 'the', 'mat'],
  ['the', 'dog', 'sat', 'on', 'the', 'log'],
  ['a', 'quick', 'brown', 'fox', 'jumps'],
  ['the', 'fox', 'sat', 'on', 'the', 'log'],
];

function computeTfIdf(corpus) {
  const N = corpus.length;
  // Build vocabulary across the corpus.
  const vocab = Array.from(new Set(corpus.flat()));
  // Document frequency for each word.
  const df = {};
  vocab.forEach((w) => {
    df[w] = corpus.reduce((acc, doc) => acc + (doc.includes(w) ? 1 : 0), 0);
  });
  // Per-doc term-frequency and tf-idf.
  const idf = {};
  vocab.forEach((w) => { idf[w] = Math.log(N / df[w]); });

  const tfPerDoc = corpus.map((doc) => {
    const counts = {};
    doc.forEach((w) => { counts[w] = (counts[w] || 0) + 1; });
    const len = doc.length;
    const tf = {};
    vocab.forEach((w) => { tf[w] = (counts[w] || 0) / len; });
    return tf;
  });

  const tfidf = tfPerDoc.map((tf) => {
    const row = {};
    vocab.forEach((w) => { row[w] = tf[w] * idf[w]; });
    return row;
  });

  return { vocab, df, idf, tfPerDoc, tfidf };
}

function SectionTfIdf() {
  const { vocab, df, idf, tfPerDoc, tfidf } = useMemo(
    () => computeTfIdf(TFIDF_CORPUS),
    []
  );
  const [hover, setHover] = useState(null); // { doc, word }
  const N = TFIDF_CORPUS.length;

  // Color scale based on max tf-idf in the matrix.
  const maxScore = useMemo(() => {
    let m = 0;
    tfidf.forEach((row) => vocab.forEach((w) => { if (row[w] > m) m = row[w]; }));
    return m || 1;
  }, [tfidf, vocab]);

  const cellColor = (v) => {
    if (v <= 0) return '#ffffff';
    const t = Math.min(1, v / maxScore);
    // Blue tint, deeper for higher scores.
    const light = 96 - Math.round(t * 50);
    const sat = 30 + Math.round(t * 50);
    return `hsl(217, ${sat}%, ${light}%)`;
  };

  const explain = (() => {
    if (!hover) {
      return (
        <span className="nlp-tfidf-empty">
          Hover any cell to see the calculation. <em>Bright</em> means the word
          is both frequent <em>in</em> that document and rare <em>across</em>{' '}
          the others — a signature term.
        </span>
      );
    }
    const { doc, word } = hover;
    const tf = tfPerDoc[doc][word];
    const id = idf[word];
    const v = tfidf[doc][word];
    return (
      <span>
        <strong>doc {doc + 1}</strong>, "<code>{word}</code>":
        {' '}tf = {tf.toFixed(3)} · idf = log({N}/{df[word]}) = {id.toFixed(3)}
        {' '}→ tf·idf = <strong>{v.toFixed(3)}</strong>
      </span>
    );
  })();

  return (
    <div className="viz-panel">
      <div className="nlp-tfidf-corpus">
        {TFIDF_CORPUS.map((doc, i) => (
          <div key={i} className="nlp-tfidf-doc">
            <span className="nlp-tfidf-doc-label">doc {i + 1}</span>
            <span className="nlp-tfidf-doc-text">{doc.join(' ')}</span>
          </div>
        ))}
      </div>

      <div className="nlp-tfidf-grid-wrap">
        <div
          className="nlp-tfidf-grid"
          style={{ gridTemplateColumns: `64px repeat(${vocab.length}, minmax(40px, 1fr))` }}
        >
          <div className="nlp-tfidf-cell nlp-tfidf-corner" />
          {vocab.map((w) => (
            <div key={`h-${w}`} className="nlp-tfidf-cell nlp-tfidf-colhead">{w}</div>
          ))}
          {tfidf.map((row, di) => (
            <React.Fragment key={di}>
              <div className="nlp-tfidf-cell nlp-tfidf-rowhead">doc {di + 1}</div>
              {vocab.map((w) => {
                const v = row[w];
                const isHi = hover && hover.doc === di && hover.word === w;
                return (
                  <div
                    key={`${di}-${w}`}
                    className={`nlp-tfidf-cell nlp-tfidf-value${isHi ? ' is-hi' : ''}`}
                    style={{ background: cellColor(v) }}
                    onMouseEnter={() => setHover({ doc: di, word: w })}
                    onMouseLeave={() => setHover(null)}
                    onClick={() => setHover({ doc: di, word: w })}
                  >
                    {v > 0 ? v.toFixed(2) : ''}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="nlp-tfidf-explain">{explain}</div>

      <p className="viz-caption">
        Words like <code>the</code> appear in every document, so their idf is
        zero and they contribute nothing. <code>quick</code>, <code>brown</code>,{' '}
        <code>fox</code>, <code>jumps</code>, <code>dog</code> light up because
        they're rare across the corpus — the signal TF-IDF was built to find.
      </p>
    </div>
  );
}

/* ============================================================
   Widget 3 — Embedding table lookup
   ============================================================ */
function SectionEmbeddings() {
  const [sentenceId, setSentenceId] = useState('she');
  const [picked, setPicked] = useState(0);
  const sentence = SENTENCES.find((s) => s.id === sentenceId);
  const pickedTok = sentence.tokens[picked];
  const vocab = Array.from(new Set(sentence.tokens));
  const COLS = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7'];

  return (
    <div className="viz-panel">
      <div className="nlp-controls">
        <label>
          Sentence
          <select
            value={sentenceId}
            onChange={(e) => { setSentenceId(e.target.value); setPicked(0); }}
          >
            {SENTENCES.map((s) => (
              <option key={s.id} value={s.id}>{s.tokens.join(' ')}</option>
            ))}
          </select>
        </label>
        <span className="nlp-controls-hint">Tap a token to look up its embedding.</span>
      </div>

      <div className="nlp-token-row">
        {sentence.tokens.map((t, i) => (
          <button
            key={i}
            className={`nlp-token-pill nlp-token-pill-button ${i === picked ? 'nlp-token-pill-active' : ''}`}
            onClick={() => setPicked(i)}
          >
            <div className="nlp-token-text">{t}</div>
            <div className="nlp-token-id">{fakeTokenId(t)}</div>
          </button>
        ))}
      </div>

      <div className="nlp-matrix-row">
        <div className="nlp-mat">
          <div className="nlp-mat-label">embedding table  (|V| × d_model)</div>
          <div
            className="nlp-mat-grid"
            style={{ gridTemplateColumns: `64px repeat(8, 44px)` }}
          >
            <div className="nlp-cell nlp-col-label" />
            {COLS.map((c) => (
              <div key={c} className="nlp-cell nlp-col-label">{c}</div>
            ))}
            {vocab.map((tok) => {
              const isHit = tok === pickedTok;
              return (
                <React.Fragment key={tok}>
                  <div className={`nlp-cell nlp-row-label${isHit ? ' nlp-row-hi' : ''}`}>{tok}</div>
                  {EMB[tok].map((v, j) => (
                    <div key={j} className={`nlp-cell${isHit ? ' nlp-trace-x' : ''}`}>
                      {v.toFixed(2)}
                    </div>
                  ))}
                </React.Fragment>
              );
            })}
          </div>
        </div>
        <div className="nlp-matrix-op">→</div>
        <div className="nlp-mat">
          <div className="nlp-mat-label">X  (one row per token)</div>
          <div
            className="nlp-mat-grid"
            style={{ gridTemplateColumns: `38px repeat(8, 42px)` }}
          >
            <div className="nlp-cell nlp-col-label" />
            {COLS.map((c) => (
              <div key={`xc-${c}`} className="nlp-cell nlp-col-label">{c}</div>
            ))}
            {sentence.tokens.map((tok, i) => {
              const isHit = i === picked;
              return (
                <React.Fragment key={`xr-${i}`}>
                  <div className={`nlp-cell nlp-row-label${isHit ? ' nlp-row-hi' : ''}`}>{tok}</div>
                  {EMB[tok].map((v, j) => (
                    <div key={`xc-${i}-${j}`} className={`nlp-cell${isHit ? ' nlp-trace-x' : ''}`}>
                      {v.toFixed(2)}
                    </div>
                  ))}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </div>

      <p className="viz-caption">
        Each token ID indexes one row of the embedding table. Stack those rows
        in sentence order and you get the matrix <Katex tex="X" />. From here on,
        every operation is linear algebra — no strings, no integers.
      </p>
    </div>
  );
}

/* ============================================================
   Widget 4 — RNN / CNN / Attention receptive-field comparison
   ============================================================ */
const ARCHES = [
  {
    id: 'rnn',
    label: 'RNN',
    receptive: (pos, T) => {
      const arr = new Array(T).fill(0);
      for (let j = 0; j <= pos; j++) arr[j] = Math.pow(0.7, pos - j);
      return arr;
    },
    parallelSteps: (T) => T,
    description:
      "Reads left-to-right, carrying a hidden state. Token i can technically reach all earlier tokens, but information about token 1 is squashed through (i−1) updates by the time it arrives — that's the long-range decay (fading colour below).",
    flaw: 'Sequential: step i waits for step i−1. T tokens → T wall-clock steps.',
  },
  {
    id: 'cnn',
    label: 'CNN (k=3)',
    receptive: (pos, T) => {
      const arr = new Array(T).fill(0);
      for (let j = 0; j < T; j++) if (Math.abs(j - pos) <= 1) arr[j] = 1;
      return arr;
    },
    parallelSteps: () => 1,
    description:
      'Each layer sees only a small fixed window (kernel size 3 → ±1 neighbour). Reaching position 1 from position 100 requires stacking many layers — the receptive field grows linearly with depth.',
    flaw: 'Parallel within a layer, but global context demands depth.',
  },
  {
    id: 'attn',
    label: 'Attention',
    receptive: (pos, T) => new Array(T).fill(1),
    parallelSteps: () => 1,
    description:
      'Every token can attend to every other token in a single layer. No decay, no kernel limit — the receptive field is the whole sequence from layer 1.',
    flaw: 'Costs O(T²) in compute and memory. Manageable up to ~thousands of tokens.',
  },
];

function SectionHistory() {
  const [archId, setArchId] = useState('attn');
  const T = 7;
  const [focus, setFocus] = useState(3);
  const arch = ARCHES.find((a) => a.id === archId);
  const intensities = arch.receptive(focus, T);

  return (
    <div className="viz-panel">
      <div className="nlp-controls">
        <div className="nlp-tabs" role="tablist">
          {ARCHES.map((a) => (
            <button
              key={a.id}
              role="tab"
              className={`nlp-tab ${archId === a.id ? 'active' : ''}`}
              onClick={() => setArchId(a.id)}
            >
              {a.label}
            </button>
          ))}
        </div>
        <span className="nlp-controls-hint">
          Tap a token to see what <strong>it</strong> can reach in <em>one</em> layer.
        </span>
      </div>

      <div className="nlp-arch-stage">
        <div className="nlp-arch-label">Receptive field (1 layer)</div>
        <div className="nlp-arch-row">
          {Array.from({ length: T }, (_, i) => {
            const intensity = intensities[i];
            const isFocus = i === focus;
            return (
              <button
                key={i}
                className={`nlp-arch-node ${isFocus ? 'nlp-arch-focus' : ''}`}
                style={{
                  background: isFocus
                    ? 'var(--post-accent)'
                    : intensity > 0
                      ? `hsl(217, ${20 + intensity * 60}%, ${94 - intensity * 50}%)`
                      : '#fff',
                  color: isFocus ? '#fff' : intensity > 0.55 ? '#fff' : 'var(--post-text)',
                }}
                onClick={() => setFocus(i)}
              >
                t{i + 1}
              </button>
            );
          })}
        </div>

        <div className="nlp-arch-label" style={{ marginTop: 22 }}>
          Sequential dependency chain (T = {T})
        </div>
        <div className="nlp-arch-row">
          {Array.from({ length: T }, (_, i) => (
            <React.Fragment key={i}>
              <div className="nlp-arch-step">{i + 1}</div>
              {i < T - 1 && (
                <div className={`nlp-arch-arrow ${archId === 'rnn' ? 'nlp-arch-arrow-on' : 'nlp-arch-arrow-off'}`}>→</div>
              )}
            </React.Fragment>
          ))}
        </div>
        <div className="nlp-arch-clock">
          Wall-clock steps to compute one layer:{' '}
          <strong>{arch.parallelSteps(T)}</strong>
          {arch.parallelSteps(T) === 1 ? ' (all positions in parallel)' : ' (one at a time)'}
        </div>

        <p className="viz-caption" style={{ marginTop: 14 }}>
          <strong>{arch.label}.</strong> {arch.description} <em>{arch.flaw}</em>
        </p>
      </div>
    </div>
  );
}

/* ============================================================
   Page
   ============================================================ */
export default function NlpHistory() {
  usePageMeta({
    title: 'From TF-IDF to Attention',
    description: 'How NLP got from counting words to attention — tokens, TF-IDF, embeddings, and the RNN/CNN limits that attention was built to escape.',
    slug: 'nlp-history',
    publishedDate: '2026-01-08',
    keywords: ['NLP', 'TF-IDF', 'word embeddings', 'transformers', 'history'],
  });

  return (
    <article className="post-2026 nlp-post">
      <div className="post-wrap">
        <div className="post-hero">
          <div className="post-series-tag">
            <span className="post-live-dot" aria-hidden="true" />Transformers · Part 1 of 3
          </div>
          <h1>From TF-IDF to Attention</h1>
          <p className="post-lede">
            <em>"The bank is on the river bank."</em> Same word, two meanings; only
            the context resolves which is which. A model that wants to understand
            language has to figure out, for every word, which other words tell it
            what this word <em>means right here</em>. The story of NLP is the
            story of how that context got better — from counting words, to
            embedding them, to letting them look at each other directly.
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

        <h2 className="reveal">What is a token?</h2>
        <p>
          Before any of the math, the text has to become numbers. A{' '}
          <strong>token</strong> is the smallest unit a model reads — sometimes
          a character, sometimes a word, most often a chunk in between. The
          tokenizer chops the input into a sequence of token IDs from a fixed
          vocabulary.
        </p>
        <p>
          Modern systems use subword tokenizers like Byte-Pair Encoding. They
          keep common words whole (<code>·is</code>, <code>·you</code>) and
          split rare ones into reusable pieces (<code>Att</code> +{' '}
          <code>ention</code>). Switch modes below and watch the same sentence
          break apart differently.
        </p>
        <SectionTokens />

        <h2 className="reveal">Counting words: bag-of-words and TF-IDF</h2>
        <p>
          The earliest text models ignored order entirely. A document was a{' '}
          <strong>bag of words</strong>: a vector counting how often each
          vocabulary item appeared. "The cat sat on the mat" and "The mat sat
          on the cat" looked identical. Useful for spam filters, hopeless for
          meaning.
        </p>
        <p>
          <strong>TF-IDF</strong> is the small improvement that kept this
          approach alive for decades. Term frequency (<em>tf</em>) measures how
          often a word appears in <em>this</em> document; inverse document
          frequency (<em>idf</em>) down-weights words that appear in
          <em> every</em> document. Multiply them and you get a score that
          highlights what makes each document distinctive.
        </p>
        <Katex
          block
          tex={'\\mathrm{tfidf}(t, d) \\;=\\; \\underbrace{\\,tf(t, d)\\,}_{\\text{how often in }d} \\;\\cdot\\; \\underbrace{\\,\\log\\!\\frac{N}{df(t)}\\,}_{\\text{rare across corpus}}'}
        />
        <p>
          Here <em>t</em> is a term, <em>d</em> a document, <em>N</em> the total
          number of documents, and <em>df(t)</em> the number of them containing{' '}
          <em>t</em>. A word like <code>the</code> appears in every document, so{' '}
          <em>log(N/df)</em> → <em>log 1</em> = 0 and its score collapses. A
          rare word like <code>fox</code> in only one doc keeps a large{' '}
          <em>log(N/df)</em>, and any document where it appears lights up.
        </p>
        <SectionTfIdf />
        <p>
          TF-IDF works because rare words carry information. But it still has
          no idea that <code>dog</code> and <code>cat</code> are similar, or
          that <code>bank</code> means two different things. Every word is its
          own independent column.
        </p>

        <h2 className="reveal">From tokens to vectors</h2>
        <p>
          The next step was to give every token a learned vector — an{' '}
          <strong>embedding</strong> — so that words with related meanings end
          up near each other in space. <code>cat</code> and <code>dog</code>{' '}
          should sit close; <code>cat</code> and <code>Tuesday</code> should
          not. Word2vec (2013) and GloVe (2014) were the first widely used
          recipes; modern systems learn embeddings end-to-end with the rest of
          the model.
        </p>
        <p>
          Concretely, the embedding table is a matrix: one row per vocabulary
          item, one column per latent dimension. Looking up a token is just
          indexing into the table. Stack the rows for a sentence and you get
          the matrix <Katex tex="X" /> — the input to everything that follows.
        </p>
        <SectionEmbeddings />

        <h2 className="reveal">Where attention came from — RNNs and CNNs first</h2>
        <p>
          Embeddings give each word a meaning, but the meaning is fixed. The
          same <code>bank</code> row goes into the model whether the next word
          is <code>river</code> or <code>account</code>. Something has to
          <em> mix</em> the surrounding context in.
        </p>
        <p>
          For most of the 2010s, two architectures did that mixing.{' '}
          <strong>RNNs</strong> read the sentence one token at a time, updating
          a hidden state. Long-range context survives in principle, but the
          signal from token 1 has to pass through every intermediate update to
          reach token 100. <strong>CNNs</strong> apply a small fixed window in
          parallel; reaching far positions means stacking many layers.
        </p>
        <SectionHistory />
        <p>
          Attention solved both limits at once. Every token can look at every
          other token <em>directly</em>, in a single layer, and all positions
          are computed in parallel. The cost is quadratic in sequence length —
          but the receptive field is the whole sequence from layer one. What
          attention actually computes is the subject of{' '}
          <a className="post-link" href="/blog/self-attention">Part 2</a>.
        </p>

        <h2 className="reveal">References</h2>
        <div className="post-refs">
          <div className="head ref-cite">Cite</div>
          <div className="head ref-title">Title</div>
          <div className="head ref-link">Link</div>

          <div className="ref-cite">Wikipedia</div>
          <div>
            <div className="ref-title">
              <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" target="_blank" rel="noreferrer">tf–idf</a>
            </div>
            <div className="ref-note">Term-frequency · inverse-document-frequency, the classic IR weighting.</div>
          </div>
          <div className="ref-link"><a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" target="_blank" rel="noreferrer">wikipedia.org</a></div>

          <div className="ref-cite">Mikolov et al. 2013</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1301.3781" target="_blank" rel="noreferrer">Efficient Estimation of Word Representations in Vector Space</a>
            </div>
            <div className="ref-note">The word2vec paper — learned dense embeddings of words.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1301.3781" target="_blank" rel="noreferrer">arxiv.org</a></div>

          <div className="ref-cite">Pennington et al. 2014</div>
          <div>
            <div className="ref-title">
              <a href="https://nlp.stanford.edu/pubs/glove.pdf" target="_blank" rel="noreferrer">GloVe: Global Vectors for Word Representation</a>
            </div>
            <div className="ref-note">Co-occurrence-based embedding learning, contemporary with word2vec.</div>
          </div>
          <div className="ref-link"><a href="https://nlp.stanford.edu/pubs/glove.pdf" target="_blank" rel="noreferrer">stanford.edu</a></div>

          <div className="ref-cite">Vaswani et al. 2017</div>
          <div>
            <div className="ref-title">
              <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Attention Is All You Need</a>
            </div>
            <div className="ref-note">The Transformer paper — replaces recurrence and convolution with attention.</div>
          </div>
          <div className="ref-link"><a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">arxiv.org</a></div>
        </div>

        <footer className="post-footer">
          <p>
            <strong>Part 1 of Transformers</strong> · Next:{' '}
            <a className="post-link" href="/blog/self-attention">Inside Self-Attention (Q, K, V, and Softmax)</a>
            {' · '}Continues with:{' '}
            <a className="post-link" href="/blog/from-attention-to-transformer">From Attention to Transformer</a>.
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
