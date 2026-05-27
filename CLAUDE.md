# berniwal.github.io — project guide

Personal site + blog (Create React App, `react-router-dom` **HashRouter**, deployed via
`gh-pages`). Posts live in `src/posts/`. Long-form interactive explainers are the
signature format; the house style for them is below and is **binding** — match it exactly.

## Repo mechanics for a new post

A post is a self-contained component registered in two places:

1. `src/posts/<Name>.jsx` + `src/posts/<Name>.css` (a pair, scoped under one root class).
2. Register in **both**:
   - `src/blogPosts.js` — add `{ slug, title, category: 'Visualizing ML', excerpt, component: '<Name>' }`.
   - `src/Blog.js` — `import <Name>` and add it to `componentRegistry`.
3. Route is `/#/blog/<slug>` (HashRouter). Verify by visiting it in the dev server.
4. Static data (pre-computed JSON the widget reads) goes in `public/data/<slug>/…` and is
   fetched at `` `${process.env.PUBLIC_URL || ''}/data/<slug>/…` ``. KaTeX is already loaded
   globally from a CDN in `public/index.html` (`window.katex`) — do not re-add it.

Run the dev server with the preview tooling (`.claude/launch.json` → config `dev`, port 3055),
**not** raw `npm start`. Screenshot at a desktop width (≥1280) before reporting done — the
preview defaults to a mobile viewport, which hides alignment bugs.

**Canonical reference for new posts:** `src/posts/MinimaxSearch.jsx` + `.css`.
This is the post-2026-redesign template — white background, Inter sans, slate
palette, blue accent, micro-animations, 1120 px column aligned with the nav.
Match its structure exactly when porting another post or starting a new one.

Older interactive posts (`VisualizingRoPE.jsx`, `VisualizingAttention.jsx`,
`VisualizingSelfImprovement.jsx`, etc.) still work but use the legacy
paper/ink/DM-Serif tokens. They're correct references for *content arc* and
*widget patterns*, not for design tokens.

---

# Interactive explainer — house style

Build an interactive blog post titled "[TITLE]" as part of an interactive-explainer series.
Match this format exactly.

## Structure & stack

- Single self-contained React component (`.jsx`) + a `.css` file pair. No new framework,
  charting lib, or animation lib — plain React (`useState`/`useMemo`/`useEffect`/`useRef`)
  + inline SVG for all visuals.
- Math via KaTeX (CDN global `window.katex` through a small `<Katex tex=… block=… />`
  wrapper). Compact 4–6 line code blocks where they earn their place; honest but minimal.
- **Sanctioned domain dependencies** (not UI libs, just rules engines that would be
  silly to rewrite): `chess.js` for the Algorithms series (Minimax / Alpha-Beta / MCTS).
  Used by `src/posts/ChessAgentPlayer.jsx`, the shared play-the-algorithm widget. Same
  spirit as KaTeX — a domain utility, not a UI framework. Don't add anything new without
  noting it here.

## Design system (reuse exactly)

**Two eras coexist.** New posts use the tokens in `MinimaxSearch.css` (white
background, Inter sans, slate palette). The legacy `--paper`/`--ink`/DM-Serif
tokens listed below are documented because the existing Visualizing-ML posts
still use them — when *editing* one of those posts, match their style; when
*writing a new post*, ignore them and use the Minimax tokens.

- **New-era tokens (MinimaxSearch.css):** white background, Inter sans
  throughout, slate-200/300 borders, slate-900 text, blue-600 accent. Headings
  *not* serif; they use Inter weight 600. No `text-align: center` on the wrap.
  All elements span the full content width.
- **Legacy tokens (Visualizing-ML posts):** `--paper #FAF8F4`, `--ink #1A1A1A`,
  single accent hue (`--accent-h: 218`), `--pos` green / `--neg` red for
  "good/bad", **serif headings, sans body, mono for numbers/code**. Container
  `max-width: 1200px; margin: 0 auto; text-align: center`; **headings centered,
  body paragraphs left-aligned** in `--ink-soft`. (A common bug is body text
  inheriting the container's `text-align: center` — override it.)
- **Components (both eras):** a white-card `.viz-panel` per widget,
  `.viz-tabs` for mode-switching, a KPI counter strip, an `InfoBox` aside for
  callouts, a `Tip` hover/tap tooltip.
- **Code blocks (`.viz-code` and markdown ` ``` `):** light tinted background only.
  Use `background: #f1f5f9` (slate-50), `color: #0f172a` (slate-900),
  `border: 1px solid #e2e8f0`, `border-radius: 10px`. **Never** ship a dark code block —
  no `#1A1F2C`, no Prism `tomorrow`/`atomDark`/`oneDark` themes. Markdown posts use the
  `oneLight` Prism theme via `src/Blog.js` with `codeTagProps={{ style: { background:
  'transparent', textShadow: 'none' } }}` — the transparent inner background is required,
  otherwise the syntax tokens render with a white halo that looks like a cursor selection.
- **Mobile-friendly:** every hover must also be tap-accessible; grids collapse to one column
  under ~720px.

## Post chrome (post-2026 redesign — binding for new posts)

The reference implementation is `MinimaxSearch.jsx` / `.css`. Copy its
structure verbatim and rename the `mm-*` prefix to a slug-scoped one
(e.g. `ab-*` for Alpha-Beta, `mcts-*` for Monte Carlo).

- **Container width:** `.<slug>-wrap` is `max-width: 1120px` with horizontal
  padding `28px` — this matches `.nav-inner` exactly so the post column ends
  at the same x-coordinate as the `Contact` link in the nav. Don't cap
  individual elements (h1, h2, p, panels, refs, footer) at a narrower width —
  let them all fill the wrap. We tried 720 / 820 and the user prefers the
  wider column.
- **Header block:** wrap series-tag + h1 + lede + byline in a `.<slug>-hero`
  div. The animations rely on it being the direct parent. Series-tag format:
  `<Series Name> · Part <N> of <M>`. Byline format: `By <author> & Claude
  (Anthropic) · co-written and co-designed · LinkedIn · GitHub`. Byline gets
  a thin `border-top` slate rule above it.
- **Sections:** every `h2` carries `className="reveal"` so the global
  IntersectionObserver in `src/App.js` fades it in as you scroll past.
- **Widgets:** all live inside `.viz-panel` (white card, 14 px radius, slate
  border, soft shadow). Panels are auto-observed by the same reveal observer —
  don't add `.reveal` to them, the App.js selector picks up
  `.<slug>-post .viz-panel:not(.in)` automatically. If you add a new post
  slug, update that selector or generalize it.
- **Code blocks:** wrap in a `<details className="<slug>-code-details">` so
  they're collapsed by default. Summary has a rotating `›` chevron icon, the
  label, and a `click to expand` hint that disappears when open. Code inside
  uses the `oneLight` Prism style (see *Design system* above).
- **References:** 3-column grid `<cite>` / `<title-with-link>` / `<link>`,
  low-chrome, left-aligned. Use the `.<slug>-refs` class with the same
  styling as `MinimaxSearch.css`.
- **Footer:** `<strong>Part N of <Series></strong> · Next: ... · Continues
  with: ...` then the bio line with LinkedIn/GitHub.

## Micro-animations (binding for new posts)

Implemented globally in `src/App.js` (`ReadingProgress` + `useRevealObserver`);
post-specific keyframes live in `MinimaxSearch.css`. Reproduce them in any new
post that uses the new chrome:

- **Top scroll-progress bar** — fixed 2 px accent bar at `top: 0; z-index: 100`.
  Already mounted in `App.js`, works on every route. Don't reimplement.
- **Hero entrance** — `.<slug>-hero > *` opacity 0 → 1 + translateY(14 → 0)
  via `@keyframes <slug>-hero-rise`, 800 ms cubic-bezier(.2,.7,.2,1),
  staggered at 80 / 180 / 320 / 460 ms for series-tag / h1 / lede / byline.
- **Live dot** — a small pulsing `<span class="<slug>-live-dot">` before the
  series-tag text. 7 px circle in accent color, box-shadow ring expands and
  fades on a 2.2 s cubic-bezier(.4,0,.2,1) loop. Use `color-mix` to fade the
  shadow without an extra opacity layer.
- **Reveal-on-scroll** — `.reveal` (on each `h2`) and `.viz-panel` /
  `.<slug>-code-details` (automatic via the observer) fade-rise on intersect.
  Reveal styles: `opacity: 0; transform: translateY(14px); transition: 700ms
  cubic-bezier(.2,.7,.2,1)`. The `.in` class flips them to visible.
- **Hover lift** — `.viz-panel.in:hover` gets a slightly stronger shadow.
  Subtle; don't translate the panel.
- **State-change pop** — when a widget reveals a "chosen" cell or path, give
  it a short scale bounce (`@keyframes mm-cell-pop`, 360 ms, max scale 1.08).
- **`prefers-reduced-motion`** — short-circuit every animation/transition:
  `animation: none !important; transition: none !important; opacity: 1
  !important; transform: none !important;`. Non-negotiable.

## Shared widget components

Some widgets are reusable across posts in a series. Put them in
`src/posts/` as their own files (e.g. `ChessAgentPlayer.jsx` is the
play-the-algorithm board reused by Minimax / Alpha-Beta / MCTS). Export both
the React component **and** the algorithm function so future posts can swap
in their own agent (e.g. `import ChessAgentPlayer, { minimaxAgent } from
'./ChessAgentPlayer'`). Cap engine search depth at 3 in vanilla minimax (UI
stays responsive); algorithms with pruning can go deeper.

## The content arc — a progression, not a flat list

- Open with a concrete hook (an italic example + a flip that changes the answer), then name
  the abstract idea, then why it matters.
- Each section is motivated as fixing the previous one's limitation. Establish an organizing
  framework (e.g. a 2-axis map) early, place each method on it as you go, and revisit it as a
  screenshot-worthy comparison table near the end.
- Build the hardest/most important widget first and verify it in the browser before the rest.

## Post length — short and focused beats long and complete (binding)

The post-2026 redesign favours **short, focused posts** over long
encyclopaedic ones. The Algorithms series (Minimax → Alpha-Beta → MCTS) is
the model: each post explains *one* algorithm, ships in roughly **600–1200
words** and **2–4 widgets**, and links to its neighbours. A reader can grasp
any single post in one sitting.

- **Default to shorter.** If the outline has more than ~4 widgets or ~1500
  words, ask yourself whether it should be two posts. The answer is almost
  always yes.
- **One idea per post.** If you're tempted to write "Part 1: history + math +
  intuition + implementation + comparison", split it. Each of those is its own
  post, and each gets to be advertised separately on the landing page.
- **Series, not mega-posts.** Use `Series · Part N of M` in the kicker and a
  `Next / Previous` footer to thread them together. Cross-link in the prose
  when one post relies on a concept defined in another.
- **The old 2300–2700-word / 6-widget targets are deprecated** for new posts.
  Older Visualizing-ML posts that already hit those targets are fine — don't
  cut them; they're a different era. But don't write new ones to match.
- **Why shorter wins:** each post advertises independently in the writing
  list. Three 800-word posts on related ideas get three writing-list entries
  (more surface area, more entry points) instead of one. Short posts also
  match the voice rules — concise, scannable, engaging.

## Voice & rhythm (binding)

The defining feature of this blog is that hard ideas feel intuitive once you can *see* them.
The prose serves the visuals, not the other way around. Keep it simple, short, light.

- **Short paragraphs.** Two to four sentences. If a paragraph is longer, it's almost always
  two ideas glued together — split it. Long paragraphs are the single most common drift from
  the house style.
- **Structure visibly.** Use H2 section headings, numbered or named steps, and short
  bullet/figure pairings so the reader can scan the page and still get the shape of the
  argument. Each section should answer one question the previous section opened.
- **Explain, don't assume.** When a term, symbol, or shorthand first appears, define it in
  one clause. Don't write "the policy gradient" without saying what it is. Err toward
  over-explaining the obvious; this site is read by people new to the topic.
- **Engaging, not chatty.** Start with a concrete example or a question, not a definition.
  Use "you" and active verbs. No "in this post we will explore" boilerplate. No filler
  ("essentially", "basically", "it should be noted that").
- **Light, not academic.** Italics over hedges. A short aside is fine; a footnote is not.
  Wit is welcome; jokes are not. The reader should feel they're walking next to a friend who
  happens to know the material — not reading a textbook.
- **Visuals carry the load.** If a paragraph repeats what a widget already shows, cut the
  paragraph. The prose between widgets is connective tissue: motivate the next thing, name
  what just happened, move on.
- **Cite when claiming.** Any quantitative claim, paper, or historical fact gets a working
  link inline. If you're not sure of a number, describe the direction instead of inventing one.

## Rigor & honesty

- Verify every quantitative claim and paper detail against the actual source before shipping;
  where a number isn't certain, **describe the direction rather than inventing one**. Cite
  every paper with a working link.
- Explain every symbol the moment it appears — never drop a formula "out of the blue." When a
  reader would reasonably object, address the objection inline.

## Endings (firm conventions)

- A **References** section at the very end: a low-chrome, paper-style 3-column list (cite /
  title-with-link / link), grouped by section, left-aligned, no widget styling. Titles are
  clickable. Include an **"Earlier posts"** group for series cross-links.
- A **footer**: "Part N — Previous: […] · Start of the series: […]". **Never promise future
  posts** anywhere.
- A **byline** crediting human + Claude as co-authors, with LinkedIn/GitHub.

## Process

1. Read the existing posts first; confirm the conventions and the progression/framing before
   writing code. Start from `MinimaxSearch.jsx` for new posts in the redesigned style.
2. Propose the outline + a one-paragraph prose sketch of each widget for approval.
3. Build the anchor widget, show it, iterate; then fill in the rest.
4. Run the dev server and screenshot before reporting done.

**Porting from a legacy markdown post:** preserve the original prose
verbatim — that text was the user's writing and they have signed off on it.
The port is strictly visual: replace static PNGs with interactive widgets,
swap to the new chrome + animations. Don't expand explanations, don't add
new sections, don't rephrase paragraphs unless the user explicitly asks.
