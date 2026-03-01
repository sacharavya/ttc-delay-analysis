# ttc-delay-analysis

Predictive analytics for TTC bus delays using XGBoost and SHAP explainability, built on stop-level AVL/GPS data from Toronto Transit Commission.

## Research Question

> **"To what extent can an XGBoost model predict TTC bus schedule deviation at the stop level, and what are the dominant contributing factors as revealed through SHAP explainability analysis?"**

**Sub-questions:**

1. What prediction accuracy is achievable for stop-level delays using operational, temporal, and meteorological features?
2. What is the relative importance of operational/temporal vs weather factors — and how does this compare to Matseliukh et al. (2025), who found 52% operational vs 45% weather?
3. How do delay patterns vary across routes, time periods, and seasons?

**Hypothesis:** Operational/temporal factors will dominate over weather, consistent with Matseliukh et al. — but Toronto's harsh winters may give weather a relatively larger role than in the European context.

## Project File Structure

```
ttc-delay-analysis/                        ← Jupyter root
├── data/
│   ├── raw/
│   │   ├── avl/
│   │   │   ├── stop_level/
│   │   │   │   ├── 29/   202501_rt29.csv … 202512_rt29.csv  (~200MB each, 12 files)
│   │   │   │   └── 39/   202501_rt39.csv … 202512_rt39.csv
│   │   │   └── trip_level/
│   │   │       └── transsee.csv  (9.1 GB — backup, not used yet)
│   │   ├── gtfs/
│   │   │   ├── GTFS_ttc_YYYY-MM-DD_HHMM/   (18 raw feed folders)
│   │   │   └── gtfs_merged_29_39/           (merged output — 12 CSVs)
│   │   └── weather/                         (not yet joined)
│   └── processed/
│       └── stop_delays/
│           ├── delays_rt29_202501.parquet … delays_rt29_202512.parquet
│           ├── delays_rt39_202501.parquet … delays_rt39_202512.parquet
│           └── delays_all_routes_2025.parquet   ← 25 MB — main output
├── notebooks/
│   ├── GTFS Handling.ipynb      ← COMPLETE
│   ├── AVL_GPS_MATCH.ipynb      ← COMPLETE
│   ├── 29/
│   │   ├── 01_data_ingestion.ipynb
│   │   ├── 02_eda.ipynb
│   │   ├── 03_feature_engineering.ipynb
│   │   └── 04_modeling.ipynb
│   └── 39/   (same structure as 29)
└── src/utils/text_to_csv.py
```

### What was done

- Discovered 18 GTFS feed folders covering Dec 2024 – Dec 2025
- Audited schema consistency, row counts, and route timelines across all feeds
- Identified that routes 29 & 39 changed on **Aug 29 2025** and **Nov 13 2025**
- Identified that the **2025-02-28 feed has a stale calendar** (covers Nov–Dec 2024 — must be excluded from calendar-based date joins)
- Merged all feeds into `gtfs_merged_29_39/` using GTFS-spec primary keys
- Deduplication strategy: identical rows dropped; conflicting rows (same PK, different data) kept with `_feed_date` tag

### GTFS Primary Keys Used

| File               | Primary Key                 |
| ------------------ | --------------------------- |
| stops.csv          | stop_id                     |
| routes.csv         | route_id                    |
| trips.csv          | trip_id                     |
| stop_times.csv     | trip_id, stop_sequence      |
| calendar.csv       | service_id                  |
| calendar_dates.csv | service_id, date            |
| shapes.csv         | shape_id, shape_pt_sequence |

### Key Findings

- Route 29 = `route_id` "29", Route 39 = `route_id` "39"
- `stop_times.csv` = 554 MB (largest file)
- Two same-date folders exist for Nov 13 2025 (`_1219` and `_1519`) — both valid
- After merge: 12 output files in `gtfs_merged_29_39/`

### Errors Encountered & Fixed

| Error                                    | Cause                                                  | Fix                                                        |
| ---------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| `Found 0 GTFS feed folders`              | Wrong path (`ttc-del` instead of `ttc-delay-analysis`) | Use `Path.home() / "Desktop" / "ttc-delay-analysis" / ...` |
| `Found 0 GTFS feed folders` (again)      | `Path("~/...")` doesn't expand tilde                   | Use `Path.home()` not `Path("~")`                          |
| `ValueError: duplicate entries` in pivot | Two folders with same date (Nov 13)                    | Replace `.pivot()` with `.groupby().sum().unstack()`       |
| `IndentationError`                       | Line continuation `\\\\\\\\` broke in copy-paste       | Split into separate variables                              |
| `NameError: routes_m`                    | Cell run out of order + typos                          | Added all definitions at cell top                          |

---

### AVL Data Schema (no header row in CSV files)

```
Col 0: date       — YYYYMMDD
Col 1: time_sec   — seconds from midnight (0–86399)
Col 2: vehicle    — vehicle number
Col 3: route      — route number
Col 4: run        — run/block number
Col 5: lat        — latitude
Col 6: lon        — longitude
Col 7: trip_id    — Trapeze internal ID (NOT GTFS trip_id)
Col 8: loading    — EMPTY / HALF_EMPTY / FULL
```

**Scale:** ~2.7M pings/month/route × 12 months × 2 routes ≈ 65M total pings

### Critical Discovery: AVL trip_id ≠ GTFS trip_id

The AVL `trip_id` (~49M range) does NOT match GTFS `trip_id` (~100M range) or `block_id` (4–5 digit numbers). These come from different TTC internal systems (Trapeze CAD/AVL vs GIRO Hastus scheduling). No public lookup table exists. **Only 4 out of 30,000 trip_ids overlap.**

This is a known, documented problem with TTC open data.

### Matching Algorithm (developed iteratively)

**Step 1 — Trip direction inference**

- Compute `lat_first` and `lat_last` for each AVL trip
- If `lat_last > lat_first` → northbound → `direction_id = 1`
- If `lat_last < lat_first` → southbound → `direction_id = 0`

**Step 2 — Find active GTFS service_ids for the date**

- Use `calendar.csv` (day-of-week + date range) + `calendar_dates.csv` (exceptions)
- Skip 2025-02-28 stale feed calendar rows

**Step 3 — Match AVL trip to GTFS trip by start time**

- For each AVL trip: find GTFS trip with same direction whose first-stop departure is closest to AVL `first_ping_sec`
- **Midnight fix:** GTFS encodes overnight trips as `>86400` sec (e.g. 24:30 = 88,200). AVL resets to 0 at midnight. For AVL trips with `first_sec < 7200`, try both raw time and `first_sec + 86400`, take whichever gives smaller diff.
- Discard matches with time diff > 5 minutes

**Result:** 98% of AVL trips matched within 2 minutes

**Step 4 — Sequence-constrained stop matching**

- For each GTFS scheduled stop (in sequence order), find AVL pings within the time window `[prev_stop_scheduled_time, next_stop_scheduled_time + 5min]`
- Among those pings, take the one spatially closest to the stop
- Discard if closest ping > 150m away
- `delay_sec = actual_time_sec − scheduled_arrival_sec`

This handles large delays because we use stop **sequence order**, not a hard time window.

### Approaches Tried and Why They Failed

| Approach                             | Problem                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------- |
| Direct trip_id join                  | AVL and GTFS use completely different ID systems. 0% overlap.                          |
| Nearest-stop spatial (no constraint) | Route 29 doubles back — stop 1 and stop 40 are spatially close. Delays of ±1200 min.   |
| ±10 min time window                  | Buses running >10 min late have no pings within the window. 65% of trips lost.         |
| First-ping time matching             | Multiple trips start near midnight with `time_sec=1,2,4` — all snap to same GTFS trip. |

### Output Dataset: `delays_all_routes_2025.parquet`

**Columns:**

| Column              | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `date`              | Service date                                          |
| `route`             | "29" or "39"                                          |
| `gtfs_trip_id`      | Matched GTFS trip_id                                  |
| `avl_trip_id`       | Original AVL trip_id (Trapeze)                        |
| `vehicle`           | Vehicle number                                        |
| `stop_id`           | GTFS stop_id                                          |
| `stop_sequence`     | Stop order within trip                                |
| `sched_arrival_sec` | Scheduled arrival (sec from midnight)                 |
| `actual_time_sec`   | Actual arrival (sec from midnight, midnight-adjusted) |
| `dist_to_stop_m`    | Distance from best AVL ping to stop (metres)          |
| `delay_sec`         | Delay in seconds (positive = late)                    |
| `delay_min`         | Delay in minutes                                      |
| `loading`           | Passenger load category                               |

**Scale:** 25 MB, full year 2025, routes 29 & 39

### Preliminary Delay Statistics (Jan 1 2025, Route 29)

- Median delay: +0.31 minutes (just over 18 seconds late)
- Mean delay: −0.76 minutes (slightly early on average — typical for TTC)
- GPS accuracy: 78% of pings within 50m of matched stop
- Match rate: 99.7% of AVL trips matched to GTFS within 5 minutes

---

## Part 3 — Visualisations Built

### Static charts (matplotlib)

- Delay distribution histogram (both routes overlaid)
- Delay by hour of day (boxplot and line chart)
- Delay by month (with route-change periods shaded)
- CDF — cumulative % on time / <5 min late
- Day-of-week pattern
- GPS accuracy (distance to matched stop)

### Interactive maps (folium)

- Stop-level average delay map (coloured circles, click for popup)
- GTFS shape vs AVL GPS trace overlay (blue = scheduled, orange = actual)
- Single-trip validation map (pings coloured by distance to shape)

---

## Known Issues & Caveats

1. **Extreme delay outliers** (±800 min) — likely wrong matches at midnight edge cases. Winsorize at ±30 min before modelling.
2. **Stale 2025-02-28 GTFS calendar** — covers Nov–Dec 2024. Filter: `calendar[calendar["_feed_date"] != "2025-02-28"]`
3. **205 stops/trip in merged GTFS** — stop_times has duplicate rows from 18 feeds. Fix: `drop_duplicates(subset=["trip_id","stop_sequence"], keep="first")` after sorting by `_feed_date` descending.
4. **No stop_id in AVL** — matching is spatial approximation (nearest stop within sequence window), not exact arrival events. Median distance: ~21m.
5. **Loading field** — three ordinal values only (no raw APC counts). Not reliable for passenger volume modelling.
6. **AVL trip matching is approximate** — ~1–2% of trips may be mismatched, especially near midnight and at route terminals.
7. **Transsee backup data** (9.1 GB) — has stop-level timestamps and GTFS-style trip_ids in ~88M range. Available as fallback if AVL matching quality is insufficient.

---
