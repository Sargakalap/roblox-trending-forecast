# Roblox Top Trending forecast

Entry for make.fun's `roblox-trending-forecast-01`. `node main.ts` reads `/input/history.ndjson` and `/input/task.json`
and writes `/output/forecast.json`: per place 1-10, a probability for each game, averaged over the 24 hourly slots.

Each game's future rank is a normal in log-rank around its latest place (with a 10% heavy tail for sudden jumps),
widening with the horizon, with a small drop-off hazard for games on the chart over ~6 days. Trend and time-of-day
terms are in the code but tuned to zero: on a backtest of 31 forecast origins (30 Oct - 8 Nov 2026) they did not help.
Each slot's game-by-place matrix is balanced so every place sums to 1. Standard library only, deterministic.
