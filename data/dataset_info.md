# SMS Spam Collection v.1 — Metadata

**Total messages:** 5160
- Ham:  4518 (87.56%)
- Spam: 642 (12.44%)

**Columns:** id, label, text  
**Format:** TSV → CSV  
**Preprocessing:**
- HTML unescaped (&lt; → <, &gt; → >)
- Unicode normalized (NFKC)
- Whitespace collapsed
- Dropped duplicates / NaNs
- Added stable IDs
- Created shuffled unlabeled copy for LLM inference