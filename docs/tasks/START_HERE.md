# Start here (team onboarding)

The core of kfactors is done: data loaders, the factor-count rules, the walk-forward engine,
metrics, the benchmark and the tests. Your job is to add **plug-ins** on top of it. You never edit
core files, and the engine only ever gives your code past data, so you cannot break the core or
leak future data by accident.

## Who does what

| Person | Card | What you build | Difficulty | Target date |
|---|---|---|---|---|
| Kamilla | [A](card_A_minvar.md) | Minimum-variance portfolio from the factor-model covariance (the **risk** evaluator) | easy | 10 Nov |
| Kamilla | [C](card_C_baselines.md) | Thin wrappers around sklearn / statsmodels / optht selectors (the **baselines**) | easy | 14 Nov |
| Adel | [B](card_B_statarb.md) | Avellaneda-Lee stat-arb trading rule on factor residuals (the **trading** evaluator) | medium | 17 Nov |
| Both | [D](card_D_docs_examples.md) | Docs site + 2 worked examples (Kamilla: factor zoo / Ken French; Adel: crypto stat-arb) | easy | 4 Dec |

Final hand-in: **Fri 18 Dec 2026**. Nursultan reviews every pull request.

## Five steps

1. **Install** (once): follow `CONTRIBUTING.md` section 2. Run the tests; everything should pass,
   and your card's tests show as *skipped* (because your part is not written yet).
2. **Read your card** in `docs/tasks/`. It explains the goal, the maths, the exact file you edit
   and the step-by-step plan. The card's section 6 has prompts you can paste into ChatGPT/Claude.
3. **Create a branch**: `git switch -c card-a-minvar` (or `card-b-statarb`, `card-c-baselines`,
   `card-d-docs`).
4. **Implement until your card's tests pass**:
   `.venv/bin/python -m pytest tests/teammates/test_card_A_minvar.py -v`
   (skipped turns into passed). Do not edit the tests; if you think a test is wrong, ask.
5. **Open a pull request** on GitHub (`CONTRIBUTING.md` section 3) and paste the checklist from
   section 7 into the description. CI runs automatically; it must be green.

## Rules that matter

* Edit only the files listed for your card. Never `git add .`
* Use AI assistants freely, but you must be able to explain every line in the final presentation.
* Small commits with clear messages (`Card A: GMV weights via np.linalg.solve`).
* Stuck for more than an hour? Write in the team chat with the error message and what you tried.

## What happens after your card is merged

The benchmark picks your strategy up automatically:
`python -m kfactors.bench real --config configs/default.yaml --out results/` then produces the
crypto stat-arb and Ken French minimum-variance k-curves. These are the remaining results of the
project.
