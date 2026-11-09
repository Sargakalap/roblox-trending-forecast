# Roblox Top Trending forecast

Entry for make.fun's `roblox-trending-forecast-01`. `node main.ts` reads `/input/history.ndjson` and `/input/task.json`
and writes `/output/forecast.json`: per place 1-10, a probability for each game, averaged over the 24 hourly slots.

Each game's future rank is a normal in log-rank around its latest place (with a 15% heavy tail for sudden jumps),
widening with the horizon. On top, shock scenarios are mixed in by the chance each has happened by the hour: a
top-15 game drops off the chart (0.1 per game per day, the rate seen in the history) and the games below move up; a
new game jumps into the top 10 (0.7 a day, mostly at #2-#7) and the games below move down; or a game that left the
top 10 in the last 72 hours comes back to its place. Trend and time-of-day terms are in the code but tuned to zero.
Backtest over 36 forecast origins (30 Oct - 9 Nov 2026): -18.2 against the house baseline's -24.0.
