# Search Console submission — walsertech.ch

Quick checklist for getting the site indexed by Google and Bing. Do once.
After deploys, the sitemap auto-refreshes — no need to re-submit unless URLs
change.

## Prerequisites

- Latest build is deployed to GitHub Pages and `https://walsertech.ch/` loads.
- `https://walsertech.ch/sitemap.xml` returns the URL list (24 entries).
- `https://walsertech.ch/robots.txt` returns a `Sitemap:` line pointing at it.

## 1. Google Search Console

1. Visit <https://search.google.com/search-console> and sign in with the
   Google account you want to use for analytics (it doesn't have to be
   the one tied to GA4 — both work).
2. Click **Add property → URL prefix** and enter `https://walsertech.ch/`.
3. Verify ownership. Easiest option: **HTML tag** — Google gives you a
   `<meta name="google-site-verification" content="...">` snippet. Paste
   it into `public/index.html` inside `<head>`, commit, push, wait for
   `gh-pages` to redeploy, then click **Verify** in Search Console.
   - Alternative: use the **Google Analytics** verification method —
     Search Console will detect the existing GA4 tag (`G-KYC3H3W1YG`)
     and verify automatically. No code change needed.
4. Once verified, go to **Sitemaps** in the left nav and submit
   `sitemap.xml`. Google should report "Success" within a few minutes.
5. Use **URL Inspection** on the homepage and on one post (e.g.
   `/blog/self-attention`) and click **Request indexing** to nudge the
   first crawl.

Indexing typically takes 3–14 days for a fresh property. Check
**Coverage → Pages** to watch URLs move from `Discovered` to `Indexed`.

## 2. Bing Webmaster Tools

1. Visit <https://www.bing.com/webmasters> and sign in.
2. **Add site** → enter `https://walsertech.ch/`.
3. Two shortcuts:
   - **Import from Google Search Console** — if you finished step 1
     above, Bing can pull the property + verification across in one
     click. Strongly recommended.
   - Otherwise: verify via XML file or meta tag, same pattern as Google.
4. Submit `https://walsertech.ch/sitemap.xml` under **Sitemaps**.
5. Bing's URL inspection tool lives under **URL Inspection**; submit a
   couple of post URLs manually to seed the crawl.

Bing's index also feeds **DuckDuckGo**, **Ecosia**, and (partially)
**ChatGPT search**, so this is worth doing even though Google ships
the bulk of traffic.

## 3. After submission

- Watch **Coverage** in both tools for crawl errors. The most common
  failure is a `Soft 404` on legacy hash-router URLs (`/#/blog/...`) —
  ignore those, the `404.html` SPA shim handles them.
- Confirm that **rich results** appear: drop any post URL into Google's
  <https://search.google.com/test/rich-results> and verify the JSON-LD
  `Article` block is detected (title, author, datePublished). That
  schema is emitted by `src/usePageMeta.js` for every post.
- Re-run a rich-results check whenever the post schema in
  `src/usePageMeta.js` changes.

## 4. Other discovery channels (low-effort)

- **Anthropic / Claude Code-style sources**: if you list the site
  somewhere (LinkedIn, GitHub profile README, ETH alumni page), those
  inbound links are the single biggest factor in how fast Google
  decides the site is worth crawling.
- **Hacker News / Lobsters**: posting one of the strongest interactive
  explainers (e.g. `/blog/self-attention`) on launch day generates a
  burst of referer traffic that often kicks off Google's crawler.
- **Twitter / Bluesky**: linking from accounts with a few followers is
  enough — both engines treat social-discovered URLs as crawl seeds.

No paid distribution needed; the interactive widgets do the work.
