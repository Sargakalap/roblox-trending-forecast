// Forecast of Roblox's Top Trending chart (computer, US): for each of the top 10 places, a probability for each game
// being there, averaged over every hourly read of the scored window. Node 22, standard library only, deterministic.
//
// Model, per game, in log-rank space (log 1 = #1):
//   level   the latest log-rank, blended with a time-weighted trend line through the recent reads;
//   drift   that trend line's slope, damped so it fades over a day;
//   hour    a per-game time-of-day offset learned from the last three days' residuals (Blox Fruits taking #1 in
//           US night hours, for example), shrunk towards zero when there is little evidence;
//   spread  a normal in log-rank whose width grows with the horizon;
//   hazard  an optional chance of leaving the chart that grows with time on it (tuned off: the shocks cover it);
//   shocks  scenarios mixed in by the chance they have happened by each hour: a top-15 game drops off the chart and
//           the games below move up (DONT SPLASH from #1, Blox Fruits from #2), a new game jumps into the top 10 and
//           the games below move down, or a game that just left the top 10 comes back to its place.
// Every slot's game-by-place matrix is balanced (Sinkhorn) so each place sums to 1 and each game to its survival,
// with a "newcomer" row taking the mass of games we cannot name. Places 1-10 averaged over the slots are the output.
//
// Outside the grader: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HOUR = 3_600_000;

export interface Params {
  /** Hours of history the trend line looks back over, and its exponential weight decay (hours). */
  trendHours: number;
  trendTau: number;
  /** Weight of the trend line's value now against the latest read (0 = latest read only). */
  levelBlend: number;
  /** Multiplier on the trend slope, the hours over which it fades, and its cap (log-rank per hour). */
  drift: number;
  driftFade: number;
  slopeCap: number;
  /** Time-of-day offset: multiplier, pseudo-count shrinkage, kernel width (hours). */
  tod: number;
  todShrink: number;
  todWidth: number;
  /** Log-rank spread: sigma^2 = s0^2 + s1^2 * hours ahead. */
  s0: number;
  s1: number;
  /** Chance of leaving the chart per day: base, plus this per day on the chart beyond ageStart days. */
  haz0: number;
  hazAge: number;
  ageStart: number;
  /** Newcomers enter towards the bottom: their place profile is exp(newTilt * (k-1)/(K-1)). */
  newTilt: number;
  /** Heavy tail: this share of each game's rank spread is wideMul times as wide, for sudden jumps and falls. */
  wideW: number;
  wideMul: number;
  /** Shock scenarios, each mixed in with the probability that it has happened by the slot's hour:
   *  dropout   a game in the top dropTop leaves the chart at dropRate per day, and every game below moves up one;
   *  entry     a new game lands in the top 10 at entryRate per day, at a place drawn from ENTRY_PLACES, and every
   *            game from that place down moves down one;
   *  return    a game that left the top 10 in the last 72 hours comes back to its old place: returnP in all,
   *            spread over time with returnTau hours, counted from when it was last seen. */
  dropRate: number;
  dropTop: number;
  entryRate: number;
  returnP: number;
  returnTau: number;
  /** Compute every slotStep-th slot (the slots between share its matrix) to keep runs fast. */
  slotStep: number;
}

/** Where games jumping into the top 10 have landed (#2-#7 in the history): weights for places 1..10. */
const ENTRY_PLACES = [0.5, 1, 1, 1, 1, 1, 0.7, 0.5, 0.3, 0.2];

// Tuned on a backtest over every read from 2026-10-30 to 2026-11-09 (36 forecast origins, each scored on the reads
// of its next 24 hours). Trend and time-of-day came out at zero there, so they are off. Dropouts are handled by the
// shock scenarios instead of the age hazard: dropRate 0.1 is the observed rate (10 top-10 games left the chart in
// ~2,400 game-hours), entryRate 0.7 the observed jump-ins (7 in 10 days).
export const DEFAULTS: Params = {
  trendHours: 36,
  trendTau: 6,
  levelBlend: 0,
  drift: 0,
  driftFade: 24,
  slopeCap: 0.08,
  tod: 0,
  todShrink: 2,
  todWidth: 1.5,
  s0: 0.06,
  s1: 0.05,
  haz0: 0,
  hazAge: 0,
  ageStart: 6,
  newTilt: 6,
  wideW: 0.15,
  wideMul: 3,
  dropRate: 0.1,
  dropTop: 15,
  entryRate: 0.7,
  returnP: 0.15,
  returnTau: 12,
  slotStep: 3,
};

export interface Read {
  t: number;
  /** Universe ids in chart order, #1 first. */
  order: number[];
  rank: Map<number, number>;
}

export interface Window {
  from: number;
  slots: number;
  slotMs: number;
}

export interface Forecast {
  probabilities: Array<{ universe_id: number; place: number; p: number }>;
  top10: number[];
}

export function parseHistory(text: string): Read[] {
  const reads: Read[] = [];
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue;
    const r = JSON.parse(line) as { at: string; ok?: boolean; top_trending?: Array<{ rank: number; universe_id: number }> | null };
    if (r.ok === false || !Array.isArray(r.top_trending) || r.top_trending.length === 0) continue;
    const order = [...r.top_trending].sort((a, b) => a.rank - b.rank).map((g) => Number(g.universe_id));
    const rank = new Map<number, number>();
    order.forEach((u, i) => { if (!rank.has(u)) rank.set(u, i + 1); });
    reads.push({ t: Date.parse(r.at), order, rank });
  }
  reads.sort((a, b) => a.t - b.t);
  return reads;
}

/** Standard normal CDF (Abramowitz-Stegun 7.1.26 via erf), accurate to ~1e-7. */
function phi(x: number): number {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/** Weighted least squares y = a + b x; returns [a, b] (b = 0 when x has no spread). */
function wls(xs: number[], ys: number[], ws: number[]): [number, number] {
  let sw = 0, sx = 0, sy = 0;
  for (let i = 0; i < xs.length; i++) { sw += ws[i]!; sx += ws[i]! * xs[i]!; sy += ws[i]! * ys[i]!; }
  if (sw <= 0) return [ys[ys.length - 1] ?? 0, 0];
  const mx = sx / sw, my = sy / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < xs.length; i++) { const dx = xs[i]! - mx; sxx += ws[i]! * dx * dx; sxy += ws[i]! * dx * (ys[i]! - my); }
  if (sxx < 1e-9) return [my, 0];
  const b = sxy / sxx;
  return [my - b * mx, b];
}

interface GameModel {
  id: number;
  level: number;
  slope: number;
  /** Residual samples for the time-of-day offset: [UTC hour of day, residual]. */
  todSamples: Array<[number, number]>;
  ageDays: number;
  /** Place on the last chart, or 0 when it is not on it. */
  lastRank: number;
  /** For a game off the last chart's top 10 that was in a top 10 in the last 72 hours: its last such place and when. */
  wasTop?: { rank: number; t: number };
}

function hourOfDay(t: number): number {
  return ((t % (24 * HOUR)) + 24 * HOUR) % (24 * HOUR) / HOUR;
}

function buildModels(reads: Read[], p: Params): { games: GameModel[]; size: number; tLast: number } {
  const last = reads[reads.length - 1]!;
  const tLast = last.t;
  const size = last.order.length;
  const missing = Math.log(size + 1);
  // Candidates: every game on the last chart, plus any game in the top 40 of a read in the last 12 hours.
  const ids: number[] = [...last.order];
  const seen = new Set(ids);
  for (const r of reads) {
    if (r.t < tLast - 12 * HOUR) continue;
    for (const u of r.order.slice(0, 40)) if (!seen.has(u)) { seen.add(u); ids.push(u); }
  }
  // Games that held a top-10 place in the last 72 hours but no longer do: candidates to come back.
  const wasTop = new Map<number, { rank: number; t: number }>();
  for (const r of reads) {
    if (r.t < tLast - 72 * HOUR) continue;
    r.order.slice(0, 10).forEach((u, i) => wasTop.set(u, { rank: i + 1, t: r.t }));
  }
  for (const u of wasTop.keys()) if (!seen.has(u)) { seen.add(u); ids.push(u); }
  const firstSeen = new Map<number, number>();
  for (const r of reads) for (const u of r.order) if (!firstSeen.has(u)) firstSeen.set(u, r.t);

  const trendReads = reads.filter((r) => r.t >= tLast - p.trendHours * HOUR);
  const todReads = reads.filter((r) => r.t >= tLast - 72 * HOUR);
  const games: GameModel[] = [];
  for (const id of ids) {
    const y = (r: Read) => { const k = r.rank.get(id); return k === undefined ? missing : Math.log(k); };
    const xs = trendReads.map((r) => (r.t - tLast) / HOUR);
    const ys = trendReads.map(y);
    const ws = trendReads.map((r) => Math.exp((r.t - tLast) / HOUR / p.trendTau));
    const [a, b] = wls(xs, ys, ws);
    const yLast = y(last);
    const level = (1 - p.levelBlend) * yLast + p.levelBlend * a;
    const slope = Math.max(-p.slopeCap, Math.min(p.slopeCap, b));
    // Time-of-day residuals against a plain line through the last 72 hours, from reads where the game is charted.
    const present = todReads.filter((r) => r.rank.has(id));
    const todSamples: Array<[number, number]> = [];
    if (present.length >= 4) {
      const px = present.map((r) => (r.t - tLast) / HOUR);
      const py = present.map(y);
      const [a2, b2] = wls(px, py, present.map(() => 1));
      present.forEach((r, i) => todSamples.push([hourOfDay(r.t), py[i]! - (a2 + b2 * px[i]!)]));
    }
    const ageDays = (tLast - (firstSeen.get(id) ?? tLast)) / (24 * HOUR);
    const lastRank = last.rank.get(id) ?? 0;
    const top = wasTop.get(id);
    const comeback = top && (lastRank === 0 || lastRank > 10) && top.t < tLast ? top : undefined;
    games.push({ id, level, slope, todSamples, ageDays, lastRank, ...(comeback ? { wasTop: comeback } : {}) });
  }
  return { games, size, tLast };
}

function todOffset(g: GameModel, hour: number, p: Params): number {
  if (g.todSamples.length === 0 || p.tod === 0) return 0;
  let sw = 0, sr = 0;
  for (const [h, res] of g.todSamples) {
    let d = Math.abs(h - hour);
    if (d > 12) d = 24 - d;
    const w = Math.exp(-(d * d) / (2 * p.todWidth * p.todWidth));
    sw += w; sr += w * res;
  }
  return (p.tod * sr) / (sw + p.todShrink);
}

/** A shock scenario: none, game `index` drops out, an unnamed game enters at `place`, or game `index` returns there. */
interface Shock {
  kind: 'none' | 'drop' | 'entry' | 'return';
  index: number;
  place: number;
}

/** A normal in log-rank around mu (plus its heavy tail) over places 1..size; returns the profile and its mass. */
function rankProfile(mu: number, sigma: number, edges: Float64Array, size: number, p: Params): [Float64Array, number] {
  const wide = sigma * p.wideMul;
  const row = new Float64Array(size);
  let prev = 0;
  for (let k = 0; k < size; k++) {
    const c = (1 - p.wideW) * phi((edges[k + 1]! - mu) / sigma) + p.wideW * phi((edges[k + 1]! - mu) / wide);
    row[k] = Math.max(c - prev, 1e-300);
    prev = c;
  }
  return [row, prev];
}

/** One slot's game-by-place probabilities under one shock (rows: the games, in order; columns: places 1..size). */
function slotMatrix(games: GameModel[], size: number, hoursAhead: number, hour: number, p: Params, shock: Shock): Float64Array[] {
  const h = Math.max(0, hoursAhead);
  const sigma = Math.sqrt(p.s0 * p.s0 + p.s1 * p.s1 * h);
  const fade = p.driftFade > 0 ? p.driftFade * (1 - Math.exp(-h / p.driftFade)) : h;
  const edges = new Float64Array(size + 1);
  for (let k = 0; k <= size; k++) edges[k] = Math.log(k + 0.5);
  const mus = games.map((g) => Math.max(0, g.level + p.drift * g.slope * fade + todOffset(g, hour, p)));
  // Apply the shock as a shift of every game's expected place.
  if (shock.kind === 'drop') {
    const gone = Math.exp(mus[shock.index]!);
    for (let i = 0; i < mus.length; i++) {
      const r = Math.exp(mus[i]!);
      if (i !== shock.index && r > gone) mus[i] = Math.log(Math.max(1, r - 1));
    }
  } else if (shock.kind === 'entry' || shock.kind === 'return') {
    for (let i = 0; i < mus.length; i++) {
      const r = Math.exp(mus[i]!);
      if (i !== shock.index && r >= shock.place - 0.5) mus[i] = Math.log(r + 1);
    }
    if (shock.kind === 'return') mus[shock.index] = Math.log(shock.place);
  }
  const rows: Float64Array[] = [];
  const target: number[] = [];
  let total = 0;
  games.forEach((g, i) => {
    if (shock.kind === 'drop' && i === shock.index) {
      rows.push(new Float64Array(size));
      target.push(0);
      return;
    }
    const [row, mass] = rankProfile(mus[i]!, sigma, edges, size, p);
    const hazard = p.haz0 + p.hazAge * Math.max(0, g.ageDays + h / 24 - p.ageStart);
    const survive = shock.kind === 'return' && i === shock.index ? 1 : Math.exp(-hazard * (h / 24));
    const want = Math.min(1, mass) * survive;
    rows.push(row);
    target.push(want);
    total += want;
  });
  // An entering game is unnamed: its row takes its place's mass without being any game we name.
  if (shock.kind === 'entry') {
    const [row, mass] = rankProfile(Math.log(shock.place), sigma, edges, size, p);
    rows.push(row);
    target.push(Math.min(1, mass));
    total += Math.min(1, mass);
  }
  // Newcomers and the games we cannot name fill whatever the named games leave, mostly near the bottom.
  const fresh = new Float64Array(size);
  for (let k = 0; k < size; k++) fresh[k] = Math.exp((p.newTilt * k) / Math.max(1, size - 1));
  rows.push(fresh);
  target.push(Math.max(1e-6, size - total));
  // Sinkhorn: rows to their targets, every place to 1.
  for (let it = 0; it < 40; it++) {
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i]!;
      let s = 0;
      for (let k = 0; k < size; k++) s += row[k]!;
      const f = s > 0 ? target[i]! / s : 0;
      for (let k = 0; k < size; k++) row[k]! *= f;
    }
    for (let k = 0; k < size; k++) {
      let s = 0;
      for (const row of rows) s += row[k]!;
      const f = s > 0 ? 1 / s : 0;
      for (const row of rows) row[k]! *= f;
    }
  }
  return rows;
}

export function forecast(reads: Read[], win: Window, places: number, p: Params = DEFAULTS): Forecast {
  const { games, size, tLast } = buildModels(reads, p);
  const n = games.length;
  const avg: Float64Array[] = games.map(() => new Float64Array(places));
  const entryTotal = ENTRY_PLACES.reduce((a, b) => a + b, 0);
  const step = Math.max(1, Math.round(p.slotStep));
  for (let s0 = 0; s0 < win.slots; s0 += step) {
    // This computed slot stands for the slots s0..s0+step-1; it is evaluated at their middle.
    const covers = Math.min(step, win.slots - s0);
    const t = win.from + (s0 + (covers - 1) / 2) * win.slotMs + 60_000;
    const h = Math.max(0, (t - tLast) / HOUR);
    const shocks: Array<[number, Shock]> = [];
    const qDrop = 1 - Math.exp((-p.dropRate * h) / 24);
    if (qDrop > 0) games.forEach((g, i) => { if (g.lastRank >= 1 && g.lastRank <= p.dropTop) shocks.push([qDrop, { kind: 'drop', index: i, place: 0 }]); });
    const qEntry = 1 - Math.exp((-p.entryRate * h) / 24);
    if (qEntry > 0) ENTRY_PLACES.forEach((w, e) => { if (e < size) shocks.push([(qEntry * w) / entryTotal, { kind: 'entry', index: -1, place: e + 1 }]); });
    games.forEach((g, i) => {
      if (!g.wasTop || p.returnP <= 0) return;
      const since = (tLast - g.wasTop.t) / HOUR;
      const q = p.returnP * (Math.exp(-since / p.returnTau) - Math.exp(-(since + h) / p.returnTau));
      if (q > 0) shocks.push([q, { kind: 'return', index: i, place: g.wasTop.rank }]);
    });
    let qAll = shocks.reduce((a, [q]) => a + q, 0);
    const scale = qAll > 0.9 ? 0.9 / qAll : 1;
    qAll *= scale;
    const mixes: Array<[number, Shock]> = [[1 - qAll, { kind: 'none', index: -1, place: 0 }], ...shocks.map(([q, sh]): [number, Shock] => [q * scale, sh])];
    for (const [w, shock] of mixes) {
      const m = slotMatrix(games, size, (t - tLast) / HOUR, hourOfDay(t), p, shock);
      const f = (w * covers) / win.slots;
      for (let i = 0; i < n; i++) for (let k = 0; k < places; k++) avg[i]![k]! += m[i]![k]! * f;
    }
  }
  const probabilities: Forecast['probabilities'] = [];
  for (let k = 0; k < places; k++) {
    // Name a game at this place only while its probability beats the share it would get unnamed.
    let named = games.map((g, i) => ({ id: g.id, p: avg[i]![k]! })).filter((x) => x.p > 1e-6);
    named.sort((a, b) => b.p - a.p || a.id - b.id);
    named = named.slice(0, 100);
    for (;;) {
      const sum = named.reduce((s, x) => s + x.p, 0);
      const share = (1 - sum) / Math.max(1, size - named.length);
      const keep = named.filter((x) => x.p > share);
      if (keep.length === named.length) break;
      named = keep;
    }
    for (const x of named) probabilities.push({ universe_id: x.id, place: k + 1, p: Math.floor(x.p * 1e7) / 1e7 });
  }
  // Best guess at the order: greedily the likeliest unused game at each place.
  const used = new Set<number>();
  const top10: number[] = [];
  for (let k = 0; k < places; k++) {
    let best = -1, bestP = -1;
    for (let i = 0; i < n; i++) {
      if (used.has(i)) continue;
      const q = avg[i]![k]!;
      if (q > bestP || (q === bestP && games[i]!.id < games[best]!.id)) { best = i; bestP = q; }
    }
    used.add(best);
    top10.push(games[best]!.id);
  }
  return { probabilities, top10 };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as {
    places: number; output: string; window: { from: string; slots: number; slot_minutes: number };
  };
  const reads = parseHistory(readFileSync(join(inputDir, 'history.ndjson'), 'utf8'));
  const win: Window = { from: Date.parse(task.window.from), slots: task.window.slots, slotMs: task.window.slot_minutes * 60_000 };
  const out = forecast(reads, win, task.places);
  writeFileSync(join(outputDir, task.output || 'forecast.json'), `${JSON.stringify(out)}\n`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
