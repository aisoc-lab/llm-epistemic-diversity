# Mathematical proofs reference material

Reference material for the proofs domain: 9 problems, each with multiple documented valid
proof strategies.

- `problems/{1..9}.md` — the 9 problem statements (from Engel's *Problem-Solving Strategies* and Wikipedia,
  see `proof_sources.csv`).
- `solutions/{1..9}.pdf` — solution PDFs, one per problem, each containing several
  of the named strategies written out.

## Licensing note on `solutions/*.pdf`

The solution PDFs are **direct input to the annotation judge**: the proof-strategy
classification judge (`src/judges/proof_classification.py`) uploads the relevant
`solutions/{i}.pdf` alongside a model's generated proof and asks GPT-5.2 to classify which
named strategy — if any — the model used, referencing the strategies written out in the PDF.

The proofs were compiled from external sources (see
`proof_sources.csv`) and are not included in this repo. Please supply your own replacement solution PDFs covering the strategies listed in
`proof_lists/*.{txt,md}` for each problem.
