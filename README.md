# Roblox Top Trending forecast

Entry for make.fun's `roblox-trending-forecast-01`. `node main.ts` reads `/input/history.ndjson` and `/input/task.json`
and writes `/output/forecast.json`: per place 1-10, a probability for each game, averaged over the 24 hourly slots.

Each game's future rank is a normal in log-rank around its latest place, moved by its recent trend (damped) and a
per-game time-of-day offset, widening with the horizon, with a drop-off hazard for games on the chart over ~6 days.
Each slot's game-by-place matrix is balanced so every place sums to 1. Standard library only, deterministic.
