#!/usr/bin/env node
// Build the Word report (tables) straight from the vllm bench JSON files in results/.
// Usage: node scripts/make_report_docx.js [repo_root] [output.docx]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, PageOrientation, BorderStyle, LevelFormat, Footer, PageNumber,
} = require("docx");

const ROOT = process.argv[2] || path.join(__dirname, "..");
const OUT = process.argv[3] || path.join(ROOT, "reports", "DeepSeek-V4-and-V4.1-Flash-benchmark-results.docx");

// ---------- data ----------
function row(d, extra) {
  const tpot = d.median_tpot_ms;
  return {
    ...extra, n: d.completed,
    ttft50: d.median_ttft_ms, ttft99: d.p99_ttft_ms,
    itl50: d.median_itl_ms, itl99: d.p99_itl_ms,
    tpot50: tpot, e2e50: d.median_e2el_ms / 1000,
    tpsUser: tpot ? 1000 / tpot : null, tpsTotal: d.output_throughput,
    acc: d.spec_decode_acceptance_rate, accLen: d.spec_decode_acceptance_length,
  };
}
function load(tag) {
  const dir = path.join(ROOT, "results", tag);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter((f) => /^in\d+_out\d+_c\d+\.json$/.test(f)).map((f) => {
    const [, inp, out, c] = f.match(/in(\d+)_out(\d+)_c(\d+)/).map(Number);
    return row(JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")), { tag, inp, out, c });
  });
}
function loadSpec(tag) {
  const dir = path.join(ROOT, "results", tag);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter((f) => /^specbench_c\d+\.json$/.test(f)).map((f) => {
    const c = Number(f.match(/_c(\d+)/)[1]);
    return row(JSON.parse(fs.readFileSync(path.join(dir, f), "utf8")), { tag, c });
  }).sort((a, b) => a.c - b.c);
}
const tp8 = load("tp8"), tp4 = load("tp4"), x2 = load("tp4x2");
const rt = loadSpec("tp4-realtext"), rtMtp = loadSpec("tp4-mtp-realtext");
const find = (rows, inp, c) => rows.find((r) => r.inp === inp && r.c === c) || {};
const findC = (rows, c) => rows.find((r) => r.c === c) || {};
const single = (rows) => rows.filter((r) => r.c === 1).sort((a, b) => a.inp - b.inp);
const v41 = load("v41-tp8");
const v41rt = loadSpec("v41-realtext"), v41ds = loadSpec("v41-dspark-realtext");
const readJson = (f) => { const p = path.join(ROOT, "results", f); return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null; };
const agents = (tag) => [1, 2, 4, 8, 16].map((n) => readJson(`${tag}/agent_s${n}.json`)).filter(Boolean);
const PRICE_GPU_H = 3.0;   // USD per GPU-hour, Akamai Cloud RTX PRO 6000 (quote)

function gpuStats(file) {
  const f = path.join(ROOT, "results", file);
  if (!fs.existsSync(f)) return null;
  let cols = null; const per = {};
  for (const line of fs.readFileSync(f, "utf8").split("\n")) {
    if (line.startsWith("#Time") || line.startsWith("# gpu") || line.startsWith("#gpu")) { cols = line.slice(1).trim().split(/\s+/); continue; }
    if (!cols || line.startsWith("#") || !line.trim()) continue;
    const v = line.trim().split(/\s+/); const r = Object.fromEntries(cols.map((c, i) => [c, v[i]]));
    const g = +r.gpu; if (Number.isNaN(g)) continue;
    (per[g] = per[g] || []).push({ pwr: +r.pwr, t: +r.gtemp, sm: +r.sm, clk: +r.pclk });
  }
  const all = Object.values(per).flat(); if (!all.length) return null;
  const busy = all.filter((x) => x.sm > 20);
  const avg = (xs, k) => xs.reduce((s, x) => s + x[k], 0) / xs.length;
  return { gpus: Object.keys(per).length, pwrBusy: avg(busy, "pwr"), pwrMax: Math.max(...all.map((x) => x.pwr)),
           tMax: Math.max(...all.map((x) => x.t)), smBusy: avg(busy, "sm"), clkBusy: avg(busy, "clk") };
}

// ---------- formatting helpers ----------
const FONT = "Arial";
const PAGE_W = 16838, MARGIN = 1080;                           // A4 landscape, 0.75" margins
const CONTENT_W = PAGE_W - 2 * MARGIN;                         // 14678 DXA
const C = { head: "1F3864", headText: "FFFFFF", band: "F2F5FA", good: "E2F0D9", bad: "FBE5D6", warn: "FFF2CC", grid: "BFC7D5" };
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "—"
  : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const k = (n) => (n >= 1000000 ? `${n / 1000000}M` : n >= 1000 ? `${n / 1000}k` : `${n}`);
const pct = (a, b) => `${a >= b ? "+" : ""}${fmt((a / b - 1) * 100)}%`;

const border = { style: BorderStyle.SINGLE, size: 4, color: C.grid };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, { header = false, fill, bold = false, align = AlignmentType.RIGHT } = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: C.head, type: ShadingType.CLEAR, color: "auto" }
      : fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: header ? AlignmentType.CENTER : align,
      children: [new TextRun({ text: String(text), font: FONT, size: 18,
        bold: header || bold, color: header ? C.headText : undefined })] })],
  });
}

// cols: [{h, w (relative), align?}], rows: [[...values]], fills?: fn(rowIdx,colIdx)->color
function table(cols, rows, { fills, firstColLeft = true } = {}) {
  const total = cols.reduce((s, c) => s + c.w, 0);
  const widths = cols.map((c) => Math.floor((c.w / total) * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((s, w) => s + w, 0);
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: cols.map((c, i) => cell(c.h, widths[i], { header: true })) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((v, ci) => cell(v, widths[ci], {
        fill: (fills && fills(ri, ci)) || (ri % 2 ? C.band : undefined),
        align: ci === 0 && firstColLeft ? AlignmentType.LEFT : (cols[ci].align || AlignmentType.RIGHT),
        bold: ci === 0,
      })) })),
    ],
  });
}

const R_ = (o) => new TextRun(typeof o === "string" ? { text: o, font: FONT } : { font: FONT, ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [R_(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [R_(t)] });
const P = (t, opts = {}) => new Paragraph({ spacing: { after: 120 }, children: [R_({ text: t, ...opts })] });
const B = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
  children: (Array.isArray(t) ? t : [t]).map(R_) });
const note = (t) => new Paragraph({ spacing: { before: 80, after: 200 },
  children: [R_({ text: t, italics: true, size: 18, color: "595959" })] });
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });
const L = AlignmentType.LEFT;

// ---------- content ----------
const body = [];
const a10 = find(tp4, 10000, 1), a120 = find(tp4, 120000, 1), a1m = find(tp4, 1000000, 1);
const e10 = find(tp8, 10000, 1);
const plateau8 = find(tp8, 10000, 128), plateau4 = find(x2, 10000, 128);
const rt1 = findC(rt, 1), mt1 = findC(rtMtp, 1), rt64 = findC(rt, 64), mt64 = findC(rtMtp, 64);

body.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [R_("DeepSeek-V4-Flash and V4.1-Flash on Akamai Cloud")] }));
body.push(P("8× NVIDIA RTX PRO 6000 Blackwell — vLLM latency, throughput and long-context benchmark", { size: 28, color: "404040" }));
body.push(P(`Test dates: 2026-10-31 (V4) and 2026-11-01 (V4.1), UTC · vLLM 0.32.0 · Generated ${new Date().toISOString().slice(0, 10)}`, { size: 18, color: "7F7F7F" }));

// 1. Executive summary
body.push(H1("1. Executive summary"));
body.push(table(
  [{ h: "Question", w: 5, align: L }, { h: "Answer", w: 9, align: L }],
  [
    ["Runs on RTX PRO 6000 (SM120) with stock vLLM?", "Yes — vLLM v0.32.0 image, no forks or patches"],
    ["Best layout on one 8-GPU node", "2 replicas × TP=4 (one per CPU socket), not one TP=8 engine"],
    ["One user, 10k prompt (TP=4)", `TTFT ${fmt(a10.ttft50 / 1000, 1)} s, ${fmt(a10.tpsUser)} tok/s`],
    ["One user, 120k prompt (TP=4)", `TTFT ${fmt(a120.ttft50 / 1000, 1)} s, ${fmt(a120.tpsUser)} tok/s`],
    ["One user, 1M-token prompt (TP=4)", `TTFT ${fmt(a1m.ttft50 / 1000)} s (~${fmt(a1m.ttft50 / 60000, 1)} min), then ${fmt(a1m.tpsUser)} tok/s`],
    ["Node peak output, 10k prompts, 128 users", `${fmt(plateau4.tpsTotal)} tok/s (2×TP=4) vs ${fmt(plateau8.tpsTotal)} tok/s (TP=8)`],
    ["MTP speculative decoding, real text", `1 user ${fmt(rt1.tpsUser)} → ${fmt(mt1.tpsUser)} tok/s (${pct(mt1.tpsUser, rt1.tpsUser)}); 64 users ${fmt(rt64.tpsTotal)} → ${fmt(mt64.tpsTotal)} tok/s (${pct(mt64.tpsTotal, rt64.tpsTotal)})`],
    ["Failed requests", "0 in ~3 h of testing"],
  ],
  { fills: (r, c) => (c === 1 && [1, 5, 6].includes(r) ? C.good : undefined) },
));
body.push(gap());
body.push(B([{ text: "Use 2 replicas × TP=4: ", bold: true },
  "the GPUs have no NVLink and sit on two CPU sockets (GPUs 0–3 and 4–7). With TP=8 every all-reduce crosses both sockets over PCIe; the GPUs average only ~270 W of 600 W. Four GPUs already hold the model (149 GB) plus 4.4 M tokens of KV cache."]));
body.push(B([{ text: "Context length barely slows generation: ", bold: true },
  `${fmt(find(tp4, 8000, 1).tpsUser)} tok/s at 8k, ${fmt(a120.tpsUser)} at 120k, still ${fmt(a1m.tpsUser)} at 1M. What grows is TTFT (prompt processing at ~9k tokens/s), so long-context sizing is driven by TTFT.`]));
body.push(B([{ text: "Enable MTP for real traffic: ", bold: true },
  "on real text the draft token is accepted ~80% of the time (1.8 tokens per step). On random-token benchmark prompts it is accepted 1–12% and slows things down — do not judge MTP on random prompts."]));
body.push(B([{ text: "DeepSeek-V4.1-Flash (section 9): ", bold: true },
  `runs on this node only with a pending vLLM fix (PR #60762 + FlashInfer 0.7.2). On that stack: 1M-token TTFT ${fmt(find(v41, 1000000, 1).ttft50 / 1000)} s (V4: ${fmt(a1m.ttft50 / 1000)} s), ${fmt(find(v41, 1000000, 1).tpsUser)} tok/s flat from 8k to 1M, ~0.55 s per coding-agent turn with prefix caching, GSM8K 245/250 and a 1M-token needle found.`]));
body.push(B([{ text: "Developer use (section 8): ", bold: true },
  "V4 is a good fit for IDE chat and code questions; for coding agents V4.1 with prefix caching is the better model (section 9.4); neither is meant for tab autocomplete."]));

// 2. Setup
body.push(H1("2. Test setup"));
body.push(table([{ h: "Item", w: 2 }, { h: "Value", w: 7, align: L }], [
  ["Platform", "Akamai Cloud — LKE (managed Kubernetes), plan g3-gpu-rtxpro6000-blackwell-8, 1 node"],
  ["GPUs", "8× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each, 600 W, compute capability 12.0 (SM120)"],
  ["GPU interconnect", "PCIe only, no NVLink; GPUs 0–3 on NUMA node 0, GPUs 4–7 on NUMA node 1"],
  ["GPU software", "Preinstalled LKE driver; NVIDIA GPU Operator with driver and toolkit disabled (Akamai guide)"],
  ["Model", "deepseek-ai/DeepSeek-V4-Flash (April 2026 release): MoE, 284 B total / 13 B active parameters, FP4 experts + FP8, 1M-token context, 149 GB on disk"],
  ["Serving engine", "vLLM 0.32.0 (vllm/vllm-openai:v0.32.0), --max-model-len 1048576, FP8 KV cache, prefix caching off, GPU memory utilization 0.90"],
  ["Layouts", "TP=8: one engine on 8 GPUs · TP=4: one engine on 4 GPUs measured alone · 2×TP=4: two engines under load at the same time"],
  ["Load generator", "vllm bench serve inside the server pod; random prompts of fixed length, 512 output tokens (--ignore-eos); MTP tests on Spec-Bench real text"],
]));
body.push(note("TTFT = time to first token · ITL = inter-token latency · TPOT = time per output token after the first · E2E = end-to-end latency · tok/s per user = 1000 / TPOT · p50 = median, p99 = 99th percentile."));

// 3. Single user
body.push(H1("3. One user, by prompt length (8k to 1M tokens)"));
const sizes = [...new Set(single(tp4).map((r) => r.inp))];
body.push(table(
  [{ h: "Prompt (tokens)", w: 2 }, { h: "TTFT TP=8 (s)", w: 2 }, { h: "TTFT TP=4 (s)", w: 2 }, { h: "Prefill TP=4 (tok/s)", w: 2 },
   { h: "tok/s per user TP=8", w: 2 }, { h: "tok/s per user TP=4", w: 2 }, { h: "ITL p50 / p99 TP=4 (ms)", w: 2 }, { h: "E2E TP=4 (s)", w: 2 }],
  sizes.map((s) => { const a = find(tp8, s, 1), b = find(tp4, s, 1);
    return [k(s), fmt(a.ttft50 / 1000, 1), fmt(b.ttft50 / 1000, 1), fmt(s / (b.ttft50 / 1000)), fmt(a.tpsUser), fmt(b.tpsUser),
            `${fmt(b.itl50, 1)} / ${fmt(b.itl99, 1)}`, fmt(b.e2e50, 1)]; }),
  { fills: (r, c) => (c === 5 ? C.good : undefined) },
));
body.push(note("512 output tokens in every case. One TP=4 engine (4 GPUs) is faster than TP=8 (8 GPUs) at every length: ~30% faster generation and slightly lower TTFT. ITL p99 within a few percent of p50: smooth streaming."));

// 4. Concurrency
body.push(H1("4. Concurrency with a 10k-token prompt"));
body.push(H2("4.1 Up to 8 users on one engine"));
body.push(table(
  [{ h: "Users", w: 1 }, { h: "TTFT TP=8 (s)", w: 2 }, { h: "TTFT TP=4 (s)", w: 2 }, { h: "tok/s per user TP=8", w: 2 },
   { h: "tok/s per user TP=4", w: 2 }, { h: "total tok/s TP=8", w: 2 }, { h: "total tok/s TP=4", w: 2 }],
  [1, 2, 4, 8].map((c) => { const a = find(tp8, 10000, c), b = find(tp4, 10000, c);
    return [c, fmt(a.ttft50 / 1000, 1), fmt(b.ttft50 / 1000, 1), fmt(a.tpsUser), fmt(b.tpsUser), fmt(a.tpsTotal), fmt(b.tpsTotal)]; }),
));
body.push(note("A single 4-GPU engine serves 8 users faster than TP=8 with twice the GPUs."));
body.push(H2("4.2 Whole node (8 GPUs): throughput plateau"));
body.push(table(
  [{ h: "Users (node)", w: 2 }, { h: "TP=8 TTFT p50 / p99 (s)", w: 3 }, { h: "TP=8 tok/s per user", w: 2 }, { h: "TP=8 total tok/s", w: 2 },
   { h: "2×TP=4 TTFT p50 / p99 (s)", w: 3 }, { h: "2×TP=4 tok/s per user", w: 2 }, { h: "2×TP=4 total tok/s", w: 2 }],
  [16, 32, 64, 128].map((c) => { const a = find(tp8, 10000, c), b = find(x2, 10000, c);
    return [c, `${fmt(a.ttft50 / 1000, 1)} / ${fmt(a.ttft99 / 1000, 1)}`, fmt(a.tpsUser, 1), fmt(a.tpsTotal),
            `${fmt(b.ttft50 / 1000, 1)} / ${fmt(b.ttft99 / 1000, 1)}`, fmt(b.tpsUser, 1), fmt(b.tpsTotal)]; }),
  { fills: (r, c) => (c === 6 ? C.good : undefined) },
));
body.push(note("2×TP=4 combined from both replicas (half the users on each; totals summed, latencies averaged). The plateau is set by prompt processing: each request brings 10,000 prompt tokens for 512 output tokens. TTFT p99 grows with the queue under overload while p50 stays ~4 s; cap concurrency at ~8–16 per replica for latency-sensitive traffic."));

// 5. Long context under load
body.push(H1("5. Long prompts under concurrency (60k and 120k)"));
body.push(table(
  [{ h: "Prompt", w: 1 }, { h: "Users (node)", w: 2 }, { h: "TTFT TP=8 (s)", w: 2 }, { h: "TTFT 2×TP=4 (s)", w: 2 },
   { h: "tok/s per user TP=8", w: 2 }, { h: "tok/s per user 2×TP=4", w: 2 }, { h: "total TP=8", w: 2 }, { h: "total 2×TP=4", w: 2 }],
  [60000, 120000].flatMap((s) => [2, 4, 8].map((c) => { const a = find(tp8, s, c), b = find(x2, s, c);
    return [k(s), c, fmt(a.ttft50 / 1000, 1), fmt(b.ttft50 / 1000, 1), fmt(a.tpsUser), fmt(b.tpsUser), fmt(a.tpsTotal), fmt(b.tpsTotal)]; })),
  { fills: (r, c) => (c === 5 ? C.good : undefined) },
));
body.push(note("With 2 users × 120k each replica handles one request alone: 13.4 s TTFT and ~91 tok/s per user, vs 21 s and 35 tok/s when both share the TP=8 engine."));

// 6. MTP
body.push(H1("6. MTP speculative decoding"));
body.push(P("DeepSeek-V4-Flash ships one multi-token-prediction (MTP) layer that drafts the next token; the main model verifies it in the same step. vLLM option: --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}. Measured on one TP=4 replica with Spec-Bench (480 real questions: chat, translation, summarization, QA, math, RAG; ~350–450-token prompts, up to 512 output tokens, same questions in both runs)."));
body.push(table(
  [{ h: "Users", w: 1 }, { h: "TPOT no MTP (ms)", w: 2 }, { h: "TPOT MTP (ms)", w: 2 }, { h: "tok/s per user no MTP", w: 2 }, { h: "tok/s per user MTP", w: 2 },
   { h: "total no MTP", w: 2 }, { h: "total MTP", w: 2 }, { h: "Gain (total)", w: 2 }, { h: "Acceptance", w: 2 }],
  rt.map((a) => { const b = findC(rtMtp, a.c);
    return [a.c, fmt(a.tpot50, 1), fmt(b.tpot50, 1), fmt(a.tpsUser), fmt(b.tpsUser), fmt(a.tpsTotal), fmt(b.tpsTotal), pct(b.tpsTotal, a.tpsTotal), b.acc != null ? `${fmt(b.acc)}%` : "—"]; }),
  { fills: (r, c) => (c === 7 ? (rt[r].c === 16 ? C.warn : C.good) : undefined) },
));
body.push(note("Random-token prompts gave only 1–12% acceptance and made MTP slower (e.g. TPOT 10.2 → 11.6 ms at 10k); that run was stopped. The 16-user point shows the same TPOT jump without MTP, so it is an engine behaviour at that batch size, not an MTP effect (not investigated). Not measured: MTP with long real documents and more than 1 draft token."));

// 7. Sizing and cost
body.push(H1("7. Sizing and cost (one node = 8 GPUs = 2 × TP=4)"));
body.push(table(
  [{ h: "Workload", w: 5, align: L }, { h: "Concurrent requests per node", w: 3 }, { h: "Evidence (measured)", w: 6, align: L }],
  [
    ["10k prompt, ≥ 25 tok/s per user, TTFT ≤ 5 s", "~16–30", `16: ${fmt(find(x2, 10000, 16).tpsUser)} tok/s, ${fmt(find(x2, 10000, 16).ttft50 / 1000, 1)} s · 32: ${fmt(find(x2, 10000, 32).tpsUser)} tok/s, ${fmt(find(x2, 10000, 32).ttft50 / 1000, 1)} s`],
    ["10k prompt, ≥ 40 tok/s per user", "~16", `16: ${fmt(find(x2, 10000, 16).tpsUser)} tok/s`],
    ["60k prompt, TTFT ≤ 10 s, ≥ 25 tok/s", "~4", `4: ${fmt(find(x2, 60000, 4).ttft50 / 1000, 1)} s, ${fmt(find(x2, 60000, 4).tpsUser)} tok/s`],
    ["120k prompt, TTFT ≤ 16 s, ≥ 25 tok/s", "2", `2: ${fmt(find(x2, 120000, 2).ttft50 / 1000, 1)} s, ${fmt(find(x2, 120000, 2).tpsUser)} tok/s · 4: ${fmt(find(x2, 120000, 4).ttft50 / 1000, 1)} s TTFT (too slow)`],
    ["1M-token prompt", "1 per replica", `TTFT ${fmt(a1m.ttft50 / 1000)} s, ${fmt(a1m.tpsUser)} tok/s`],
    ["Short chat (~400-token prompts), MTP on", "~64 per replica", `${fmt(mt64.tpsUser)} tok/s per user, ${fmt(mt64.tpsTotal)} tok/s per replica`],
  ],
  { firstColLeft: true },
));
body.push(gap());
const nodeH = 8 * PRICE_GPU_H;
const perM10k = nodeH / (plateau4.tpsTotal * 3600 / 1e6), perMchat = nodeH / (2 * mt64.tpsTotal * 3600 / 1e6);
body.push(table(
  [{ h: "Cost reference ($3 per GPU-hour)", w: 5, align: L }, { h: "Node $/hour", w: 2 }, { h: "Output tok/s (node)", w: 2 }, { h: "$ per million output tokens", w: 3 }, { h: "Basis", w: 4, align: L }],
  [
    ["10k prompts, saturated (128 users)", `$${fmt(nodeH, 2)}`, fmt(plateau4.tpsTotal), `~$${fmt(perM10k, 1)}`, "Measured; prompt processing included in the same hour"],
    ["Short chat with MTP, 64 users per replica", `$${fmt(nodeH, 2)}`, `~${fmt(2 * mt64.tpsTotal)}`, `~$${fmt(perMchat, 1)}`, "Estimate: 2 × one measured replica"],
  ],
));
body.push(note("GPU cost only (no storage, egress or idle time). A saturated node is a best case; real traffic has idle periods. Compare with the API price list for the same model before deciding."));

// 8. Developer workloads
body.push(H1("8. Fit for developer workloads"));
body.push(P("DeepSeek-V4-Flash is a general-purpose model (chat, reasoning, tool use), not a code-only model, but coding and agentic coding are among its strongest uses: DeepSeek reports SWE-bench Verified scores in the high 70s for V4-Flash with thinking on (check the official model card). Its 1M-token context and the flat decode speed measured here suit large codebases. What matters for developers is which workload they run:"));
body.push(table(
  [{ h: "Developer workload", w: 4, align: L }, { h: "Typical request", w: 3, align: L }, { h: "What the test shows", w: 6, align: L }, { h: "Fit", w: 2, align: AlignmentType.CENTER }],
  [
    ["Chat in the IDE, explain / write / review code", "1k–10k prompt, short answer", `TTFT ~${fmt(a10.ttft50 / 1000, 1)} s, ${fmt(a10.tpsUser)} tok/s (${fmt(mt1.tpsUser)} with MTP on real text); ~16–30 concurrent requests per node`, "Good"],
    ["Coding agent (multi-step edits, tool calls, terminal)", "30k–120k context per turn, many turns", `Without prefix caching every turn pays full TTFT: ${fmt(find(tp4, 32000, 1).ttft50 / 1000, 1)} s at 32k, ${fmt(find(tp4, 60000, 1).ttft50 / 1000, 1)} s at 60k, ${fmt(a120.ttft50 / 1000, 1)} s at 120k; ~4 concurrent 60k requests per node`, "Viable, needs test"],
    ["Whole-repository analysis, large refactors, audits", "240k–1M prompt", `TTFT ${fmt(find(tp4, 240000, 1).ttft50 / 1000)} s at 240k, ${fmt(a1m.ttft50 / 1000)} s at 1M; then ${fmt(a1m.tpsUser)}–${fmt(find(tp4, 240000, 1).tpsUser)} tok/s`, "Batch / async"],
    ["Tab autocomplete (fill-in-the-middle)", "< 2k prompt, < 200 ms target", "Not measured; a 284 B model with reasoning is the wrong tool — use a small code model (1–8 B) next to it", "Not a fit"],
  ],
  { fills: (r, c) => (c === 3 ? [C.good, C.warn, C.warn, C.bad][r] : undefined) },
));
body.push(gap());
body.push(H2("8.1 How many developers per node"));
body.push(P("The tests measure concurrent requests, not developers. A developer using chat has a request in flight only part of the time; a coding agent keeps one in flight almost continuously. The ratios below are assumptions to adjust with real usage data, not measurements."));
body.push(table(
  [{ h: "Usage pattern", w: 5, align: L }, { h: "Concurrent requests per node (measured)", w: 4 }, { h: "Assumed share of developers active at peak", w: 4 }, { h: "Developers per node (estimate)", w: 3 }],
  [
    ["IDE chat / Q&A, ~10k prompts", "16–30", "10–20%", "~80–300"],
    ["Coding agents, ~60k context, no prefix caching", "~4", "~100% (one agent = one request)", "~4 agents running at once"],
    ["Coding agents, ~120k context, no prefix caching", "2", "~100%", "~2 agents running at once"],
  ],
  { fills: (r, c) => (c === 3 ? (r === 0 ? C.good : C.warn) : undefined) },
));
body.push(note("Agent numbers are a worst case: prefix caching was disabled on purpose in this benchmark so every prompt was fully processed. In an agent session most of each turn's prompt repeats the previous turn, so with prefix caching on TTFT per turn should drop sharply and agents per node should rise — measured on V4.1 in section 9.4: ~0.55 s per turn up to 16 agents."));
body.push(H2("8.2 What to test before offering it to developers (prefix caching: done on V4.1, section 9.4)"));
body.push(B([{ text: "Prefix caching on, with agent traces: ", bold: true }, "replay real multi-turn sessions of a coding agent (Claude Code-style, OpenCode, Cline, Continue) against the OpenAI-compatible endpoint; measure TTFT per turn and agents per node."]));
body.push(B([{ text: "Tool calling: ", bold: true }, "enable vLLM's DeepSeek tool-call parser and confirm the agent clients parse calls correctly in thinking and non-thinking modes."]));
body.push(B([{ text: "MTP on code: ", bold: true }, "acceptance on code is usually high; confirm the +35–47% seen on Spec-Bench and try 2 draft tokens."]));
body.push(B([{ text: "Quality on the team's own tasks: ", bold: true }, "a small set of real tickets or a SWE-bench subset; also compare the -0731 checkpoint, which this test did not cover."]));
body.push(B([{ text: "Production layout: ", bold: true }, "at least two GPU nodes behind a load balancer for availability; cap concurrency per replica (~8–16) and queue the rest; authentication in front of the endpoint."]));

// 9. DeepSeek-V4.1-Flash
if (v41.length) {
  body.push(H1("9. DeepSeek-V4.1-Flash on the same node"));
  body.push(P("V4.1-Flash (released 2026-09-10, update 1017 on 2026-10-17) is the model the DeepSeek API serves today: 552 B backbone + 196 B Engram memory, 8 B active parameters in prefill and 16 B in decode, ~4× smaller KV cache per token, text and image input, DSpark speculative decoding. The 510 GB checkpoint only fits with TP=8."));
  body.push(H2("9.1 Bring-up: stock vLLM fails on SM120, a pending fix works"));
  body.push(table([{ h: "Step", w: 4, align: L }, { h: "Result", w: 10, align: L }], [
    ["Stock vLLM 0.32.0, TP=8", "Crash at start-up: DeepGEMM assertion 'block_kv == 32 or block_kv == 64'. vLLM declares 128-token KV blocks for V4.1 on every GPU except Hopper; the SM120 kernels accept 32/64 only. Known upstream (vLLM #59203, #56700)."],
    ["Fix applied", "vLLM PR #60762 (4 lines: 64-token blocks on SM12x) + FlashInfer 0.7.2.post1, installed in the stock image at container start; node not modified. Not in a vLLM release yet."],
    ["Start-up", "Download 476 GiB in 318 s; server ready in ~5 min; 47.6 GiB weights per GPU + 11.8 GiB Engram tables per rank in host RAM; KV cache 10.7 M tokens (V4 TP=8: 6.26 M)."],
    ["Quality check", (() => { const g = readJson("v41-quality/gsm8k.json"), n = readJson("v41-quality/needle.json"), gd = readJson("v41-dspark-quality/gsm8k.json");
      return `GSM8K ${g ? `${g.correct}/${g.n}` : "—"} (with DSpark ${gd ? `${gd.correct}/${gd.n}` : "—"}); two facts hidden in documents of ${n ? n.results.map((r) => `${fmt(r.prompt_tokens / 1000)}k`).join(", ") : "—"} tokens: ${n && n.results.every((r) => r.needle_50pct && r.needle_80pct) ? "all found" : "check"}.`; })()],
  ], { fills: (r, c) => (c === 1 ? [C.bad, C.warn, undefined, C.good][r] : undefined) }));
  body.push(H2("9.2 One user, by prompt length"));
  body.push(table(
    [{ h: "Prompt", w: 2 }, { h: "V4.1 TTFT (s)", w: 2 }, { h: "V4.1 tok/s", w: 2 }, { h: "V4 TP=8 TTFT (s)", w: 2 }, { h: "V4 TP=8 tok/s", w: 2 }, { h: "V4 TP=4 TTFT (s)", w: 2 }, { h: "V4 TP=4 tok/s", w: 2 }],
    single(v41).map((r) => { const a = find(tp8, r.inp, 1), b = find(tp4, r.inp, 1);
      return [k(r.inp), fmt(r.ttft50 / 1000, 1), fmt(r.tpsUser), fmt(a.ttft50 / 1000, 1), fmt(a.tpsUser), fmt(b.ttft50 / 1000, 1), fmt(b.tpsUser)]; }),
    { fills: (r, c) => (c === 1 ? C.good : undefined) },
  ));
  body.push(note("V4.1 processes prompts at ~13k tok/s (120k) and ~11k tok/s (1M) vs ~9k and ~4.6k for V4, and its generation speed does not drop with context. V4 on 4 GPUs is still faster per user at short context."));
  body.push(H2("9.3 Concurrency, 10k prompts (whole node)"));
  body.push(table(
    [{ h: "Users", w: 1 }, { h: "V4.1 TTFT p50 (s)", w: 2 }, { h: "V4.1 tok/s per user", w: 2 }, { h: "V4.1 total", w: 2 }, { h: "V4 TP=8 total", w: 2 }, { h: "V4 2×TP=4 total", w: 2 }],
    [1, 2, 4, 8, 16, 32, 64, 128].map((c) => { const r = find(v41, 10000, c), a = find(tp8, 10000, c), b = find(x2, 10000, c);
      return [c, fmt(r.ttft50 / 1000, 1), fmt(r.tpsUser, 1), fmt(r.tpsTotal), fmt(a.tpsTotal), b.tpsTotal ? fmt(b.tpsTotal) : "—"]; }),
  ));
  body.push(note("Without prefix caching, many concurrent cold long prompts still favour V4 as two independent 4-GPU engines (more tokens per user); see V4.1-FLASH.md §7.3 for 60k/120k."));
  const ag = agents("v41-agent-prefix");
  if (ag.length) {
    body.push(H2("9.4 Coding-agent simulation with prefix caching"));
    body.push(P("Each agent session: 10 turns; turn 1 sends 30k tokens, each next turn re-sends the conversation + the previous 256-token answer + 6k new tokens (tool output), up to 86k tokens. Prefix caching on, so only the new tokens are processed."));
    body.push(table(
      [{ h: "Parallel agents", w: 2 }, { h: "TTFT turn 1, cold (s)", w: 3 }, { h: "TTFT later turns p50 (s)", w: 3 }, { h: "TTFT later turns p90 (s)", w: 3 }, { h: "tok/s per agent", w: 2 }, { h: "Turns per minute", w: 2 }],
      ag.map((d) => [d.sessions, fmt(d.ttft_first_turn_p50, 1), fmt(d.ttft_later_turns_p50, 2), fmt(d.ttft_later_turns_p90, 2), fmt(1000 / d.tpot_p50_ms), fmt(d.turns_per_min)]),
      { fills: (r, c) => (c === 2 ? C.good : undefined) },
    ));
    body.push(note("Without prefix caching an 86k-token turn would cost ~6.4 s of TTFT. Random-token content: measures latency and caching, not answer quality."));
  }
  if (v41ds.length) {
    body.push(H2("9.5 DSpark speculative decoding on real text (Spec-Bench)"));
    body.push(table(
      [{ h: "Users", w: 1 }, { h: "tok/s per user, no spec", w: 2 }, { h: "tok/s per user, DSpark", w: 2 }, { h: "total, no spec", w: 2 }, { h: "total, DSpark", w: 2 }, { h: "Gain (total)", w: 2 }, { h: "Acceptance", w: 2 }, { h: "Tokens per step", w: 2 }],
      v41rt.map((a) => { const b = findC(v41ds, a.c);
        return [a.c, fmt(a.tpsUser), fmt(b.tpsUser), fmt(a.tpsTotal), fmt(b.tpsTotal), pct(b.tpsTotal, a.tpsTotal), b.acc != null ? `${fmt(b.acc)}%` : "—", fmt(b.accLen, 2)]; }),
      { fills: (r, c) => (c === 5 ? C.good : undefined) },
    ));
    body.push(note("--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":3}. V4 on one 4-GPU replica with MTP on the same questions: 161 tok/s (1 user), 1,527 tok/s (64 users)."));
  }
  body.push(H2("9.6 V4.1 or V4 for developers on this node"));
  body.push(table(
    [{ h: "Workload", w: 4, align: L }, { h: "Better choice", w: 3, align: L }, { h: "Why (measured)", w: 7, align: L }],
    [
      ["Coding agents, multi-turn, 30k–100k+ context", "V4.1 + prefix caching + DSpark", "~0.55 s per turn up to 16 agents; flat 78 tok/s decode (~155 with DSpark on real text); 10.7 M KV tokens"],
      ["Very long documents / whole repositories", "V4.1", "TTFT 18 s at 240k and 91 s at 1M (V4: 30 s and 217 s); 1M needle correct"],
      ["IDE chat, short prompts, many users", "V4, 2 × TP=4 + MTP", "~2× the node throughput from two independent engines; 98–161 tok/s per user"],
      ["Production timing", "V4 today; V4.1 after a vLLM release with PR #60762", "V4.1 runs on SM120 only with an unreleased fix; re-run the GSM8K and needle checks on the release"],
    ],
    { fills: (r, c) => (c === 1 ? [C.good, C.good, C.warn, C.warn][r] : undefined) },
  ));
}

// 10. GPU telemetry & infra
body.push(H1("10. GPU telemetry and infrastructure observations"));
const tel = [["TP=8 — all tests", "tp8/gpu-dmon.log"], ["2×TP=4 — replica A (driver of the load)", "gpu/dmon-tp4-replica-a.log"], ["2×TP=4 — replica B", "gpu/dmon-tp4-replica-b.log"]]
  .map(([n, f]) => [n, gpuStats(f)]).filter(([, s]) => s);
if (tel.length) body.push(table(
  [{ h: "Run", w: 4 }, { h: "GPUs sampled", w: 2 }, { h: "Power avg busy (W)", w: 2 }, { h: "Power max (W)", w: 2 },
   { h: "Temp max (°C)", w: 2 }, { h: "SM util busy (%)", w: 2 }, { h: "SM clock busy (MHz)", w: 2 }],
  tel.map(([n, s]) => [n, s.gpus, fmt(s.pwrBusy), fmt(s.pwrMax), fmt(s.tMax), fmt(s.smBusy), fmt(s.clkBusy)]),
));
body.push(gap());
body.push(table([{ h: "Observation", w: 3, align: L }, { h: "Detail", w: 9, align: L }], [
  ["GPU power under TP=8", "100% SM busy but ~270 W average of 600 W: the GPUs mostly wait on the all-reduce over PCIe"],
  ["NUMA placement", "Kubernetes gave each TP=4 replica the 4 GPUs of one socket (checked with nvidia-smi topo -m in both pods)"],
  ["Start-up time", "~7.5 min for one engine (weights 30 s + JIT / CUDA graphs ~6 min); 16–17 min for two replicas starting together (both read 149 GB from local disk)"],
  ["KV cache capacity", "TP=8: 6.26 M tokens · TP=4: 4.37 M tokens per replica (8.7 M per node) — memory is not the limit, prompt processing is"],
  ["Model download", "149 GB from Hugging Face in 126 s"],
  ["Stability", "No errors, restarts or failed requests during ~3 h of tests"],
]));

// 11. Difficulties
body.push(H1("11. Difficulties and fixes"));
body.push(table([{ h: "Problem", w: 4, align: L }, { h: "Cause", w: 5, align: L }, { h: "Fix", w: 5, align: L }], [
  ["kubectl: TLS handshake timeout", "Local corporate proxy answered DNS for the LKE API with a fake IP", "Pin the real API IP with --server and --tls-server-name"],
  ["Doubt whether V4-Flash runs on SM120", "Mid-2026 reports needed forks / patches (DeepGEMM unsupported architecture)", "Checked vLLM v0.32.0: native support, no patches"],
  ["Benchmarks dying with the laptop connection", "Long kubectl exec sessions", "Run the benchmark with nohup inside the pod and poll the log"],
  ["Slow start of two replicas (16–17 min)", "Both read 149 GB from the same local disk", "Documented; start replicas one at a time if start-up time matters"],
  ["TP=8 slower than expected", "No NVLink + all-reduce across two CPU sockets", "2 replicas × TP=4"],
  ["MTP slower than no MTP", "Random-token prompts: draft accepted 1–12%", "Measure MTP on real text (Spec-Bench): 80% acceptance"],
  ["Spec-Bench client errors (pandas, chat template)", "Missing package in the image; V4 ships its prompt encoder as code", "pip install pandas; --tokenizer-mode deepseek_v4 on the client"],
  ["Cleanup: crictl not found", "LKE nodes ship ctr, not crictl", "ctr -n k8s.io images rm …"],
  ["V4.1: engine dies at start-up (block_kv assertion)", "vLLM 0.32.0 declares 128-token KV blocks for V4.1 on SM120", "vLLM PR #60762 + FlashInfer 0.7.2.post1 at container start"],
  ["V4.1: FlashInfer jit-cache import error", "Per-arch sm120f wheel needs the matching jit-cache shim; no 0.7.x cubin release", "Install shim + sm120f wheel from the cu130 index; drop the old cubin"],
  ["Agent simulation: LineTooLong with 8 sessions; cold turns hitting the cache", "aiohttp 512 KB line limit; same random seed across runs", "Larger read buffer; one seed per run and session, runs repeated"],
]));

// 12. Method
body.push(H1("12. Methodology notes"));
body.push(B("Prefix caching disabled so every prompt is fully processed (repeated prompts would otherwise show artificially low TTFT)."));
body.push(B("Fixed-length random prompts and fixed 512-token outputs (--ignore-eos), so metrics are comparable across prompt sizes. MTP measured on real text (Spec-Bench) because random text defeats the draft."));
body.push(B("Benchmark client inside the server pod: results exclude client-to-server network latency."));
body.push(B("Single-node cluster: results represent one 8-GPU server. Production should use at least two GPU nodes."));
body.push(B("V4.1 quality was checked before any timing (GSM8K 250 questions, double-needle at 108k / 534k / 1.01M tokens) because of reported silent corruption on SM120."));
body.push(B("Everything the tests created on the cluster (namespace, 149 GB + 476 GiB of weights, FlashInfer wheels, container images) was deleted at the end; the GPU Operator was kept."));
body.push(note("Source data, manifests and scripts: github.com/ricardolins/akamai-cloud-deepseek-v4-flash (results/, k8s/, scripts/, docs/)."));

// ---------- document ----------
const doc = new Document({
  creator: "Akamai Cloud benchmark",
  title: "DeepSeek-V4-Flash and V4.1-Flash on 8x RTX PRO 6000 Blackwell — results",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 44, bold: true, color: C.head, font: FONT }, paragraph: { spacing: { after: 80 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: C.head, font: FONT }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E5597", font: FONT }, paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE },
      margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      R_({ text: "DeepSeek-V4-Flash on Akamai Cloud — page ", size: 16, color: "7F7F7F" }),
      R_({ children: [PageNumber.CURRENT], size: 16, color: "7F7F7F" })] })] }) },
    children: body,
  }],
});

fs.mkdirSync(path.dirname(OUT), { recursive: true });
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${buf.length} bytes); v41=${v41.length} agents=${agents("v41-agent-prefix").length} tp8=${tp8.length} tp4=${tp4.length} tp4x2=${x2.length} realtext=${rt.length}/${rtMtp.length} gpu=${tel.length}`); });
