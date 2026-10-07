/**
 * SWA Node API backend for the redaction demo.
 * POST /api/redact  ->  body: { raw_queries: string[] , browsing_history?: string[] }
 * Returns masked JSON mirroring src/privacy_core's PrivacyRedactor demo behavior.
 */
const NOISE = [
  /^https?:\/\//i, /youtube\.com/i, /login/i, /homepage/i,
  /translator/i, /^google$/i, /^facebook$/i, /^mail$/i, /^\w+\.\w+$/i,
];

function maskQuery(query, category, ctx) {
  const lower = query.toLowerCase().trim();
  for (const re of NOISE) if (re.test(lower)) return null; // filter as noise
  ctx[category] = (ctx[category] || 0) + 1;
  const n = String(ctx[category]).padStart(3, '0');
  return { token: `QUERY_${category}_${n}` };
}

module.exports = async function (context, req) {
  context.res = { headers: { 'Content-Type': 'application/json' } };
  let data;
  try {
    data = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
  } catch (e) {
    context.res = { status: 400, body: JSON.stringify({ detail: 'Invalid JSON body' }) };
    return;
  }
  const ctx = {};
  const out = {};
  try {
    if (Array.isArray(data.raw_queries)) {
      const q = [];
      for (const item of data.raw_queries) {
        if (typeof item === 'string') {
          const t = maskQuery(item, 'QUERY', ctx);
          if (t) q.push(t);
        }
      }
      if (q.length) out.queries = q;
    }
    if (Array.isArray(data.browsing_history)) {
      const b = [];
      for (const item of data.browsing_history) {
        if (typeof item === 'string') {
          const t = maskQuery(item, 'BROWSING', ctx);
          if (t) b.push(t);
        }
      }
      if (b.length) out.browsing = b;
    }
    context.res.body = JSON.stringify(out, null, 2);
    context.res.status = 200;
  } catch (err) {
    context.res = { status: 500, body: JSON.stringify({ detail: String(err) }) };
  }
};
