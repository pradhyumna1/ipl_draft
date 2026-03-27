# IPL Draft Tool

WARNING: I AM NOT TOO SURE ABOUT THE SANITY OF THE DATA, I DID THIS IN A COUPLE OF HOURS. ITS JUST A TOOL. USE YOUR HEAD

Helps you pick the best team in a snake draft using player stats. 

---

## What `draft_assistant.py` does

- Loads player stats from `player_stats_updated.csv`
- Calculates a score for each player based on:
  - batting (runs, boundaries, consistency)
  - bowling (wickets, economy, workload)
  - fielding (minor impact)
- Adjusts for:
  - consistency vs upside
  - player role (batter, bowler, all-rounder)
  - positional scarcity

It then lets you:

- view best available players
- draft players interactively
- build a valid team with constraints:
  - max 4 overseas players
  - minimum batters, bowlers, and all-rounders

---

## Files
player_stats_updated.csv # main dataset used for rankings
draft_assistant.py # draft tool
SMAT.py # optional stats updater

## How to use

### 1. (Optional) Update stats

python3 SMAT.py


### 2. Run draft assistant

python3 draft_assistant.py


---

## Commands inside the draft tool


best # show best available players
pick <name> # draft a player
team # view your current team
remove <name> # undo a pick
reset # restart draft


---

## What it optimizes for

- total fantasy points
- balanced team composition
- timing picks in a snake draft

---

## Example


draft> best
draft> pick Virat Kohli
draft> pick Jasprit Bumrah

