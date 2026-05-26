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

Canonical examples to read first and imitate: `src/posts/VisualizingSelfImprovement.jsx`,
`VisualizingRoPE.jsx`, `VisualizingAttention.jsx` (+ their `.css`).

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

## Design system (reuse exactly)

- **Tokens:** `--paper #FAF8F4`, `--ink #1A1A1A`, single accent hue (`--accent-h: 218`),
  `--pos` green / `--neg` red for "good/bad", **serif headings, sans body, mono for
  numbers/code**.
- **Layout convention:** container `max-width: 1200px; margin: 0 auto; text-align: center`;
  **headings centered, body paragraphs left-aligned** in `--ink-soft`. (A common bug is body
  text inheriting the container's `text-align: center` — override it.)
- **Components:** a white-card `.viz-panel` per widget, `.viz-tabs` for mode-switching, a
  KPI counter strip, an `InfoBox` aside for callouts, a `Tip` hover/tap tooltip.
- **Mobile-friendly:** every hover must also be tap-accessible; grids collapse to one column
  under ~720px.

## The content arc — a progression, not a flat list

- Open with a concrete hook (an italic example + a flip that changes the answer), then name
  the abstract idea, then why it matters.
- Each section is motivated as fixing the previous one's limitation. Establish an organizing
  framework (e.g. a 2-axis map) early, place each method on it as you go, and revisit it as a
  screenshot-worthy comparison table near the end.
- **At least 6 distinct interactive widgets** carrying the explanatory weight. Build the
  hardest/most important one first and verify it in the browser before the rest.
- Prose target **~2300–2700 words**; interactives do the heavy lifting.

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
   writing code.
2. Propose the outline + a one-paragraph prose sketch of each widget for approval.
3. Build the anchor widget, show it, iterate; then fill in the rest.
4. Run the dev server and screenshot before reporting done.
