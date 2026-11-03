# Contributing to kfactors

This guide is for team members working on a **task card** (`docs/tasks/card_*.md`). No prior
experience with open-source workflows is assumed. If something here is unclear, ask in your pull
request: that is what pull requests are for.

## 1. Who edits what

kfactors has a **core**, maintained by the core team, and four **contributor cards**. To avoid
conflicts, every file has exactly one owner (see the ownership table in `docs/design.md`).

| Card | You may edit | Tests that must pass |
|---|---|---|
| A: minimum-variance strategy | `src/kfactors/strategies/minvar.py` | `tests/teammates/test_card_A_minvar.py` |
| B: Avellaneda-Lee stat-arb | `src/kfactors/strategies/statarb.py` | `tests/teammates/test_card_B_statarb.py` |
| C: library baselines | `src/kfactors/baselines.py` | `tests/teammates/test_card_C_baselines.py` |
| D: docs site and examples | `mkdocs.yml`, `docs/**` (not `docs/design.md`, not `docs/tasks/`), `examples/**`, new robustness config files in `configs/` (e.g. `configs/french_robustness.yaml`; not `ci.yaml` / `default.yaml`), adding `site/` to `.gitignore` | `mkdocs build --strict`, the examples run |

**Never edit**:
* core files: everything else in `src/kfactors/`, e.g. `model.py`, `engine.py`, `rules/`,
  `data/`;
* the acceptance tests in `tests/teammates/`;
* `pyproject.toml`, `docs/design.md`, `docs/tasks/`;
* the core benchmark configs `configs/ci.yaml` and `configs/default.yaml` (the tests use them).

If you think a core file or a test is wrong, say so in your PR or open an issue. The core team
will fix it. Your code may only use the **public** core API described in `docs/design.md` and
in the docstrings.

## 2. One-time setup

```bash
git clone https://github.com/nabdullaev/kfactors.git
cd kfactors
python3.11 -m venv .venv                   # Python 3.11 or newer
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q              # should end with "passed" (card tests are skipped)
```

On Windows replace `.venv/bin/python` with `.venv\Scripts\python`. Always run Python through
the project's `.venv`, so that everyone uses the same package versions.

## 3. The branch / pull-request workflow

Never commit directly to `main`. Every card is developed on its own branch and merged through a
pull request (PR) that a core team member reviews.

```bash
# 1. start from an up-to-date main
git switch main
git pull

# 2. create your branch (one per card)
git switch -c card-a-minvar               # card-b-statarb, card-c-baselines, card-d-docs

# 3. work, test, and commit small steps with clear messages
git status                                # what changed?
git diff                                  # how did it change?
git add src/kfactors/strategies/minvar.py # add ONLY your card's files, never `git add .`
git commit -m "Card A: GMV weights via np.linalg.solve"

# 4. push the branch and open a PR on the repository web page ("Compare & pull request")
git push -u origin card-a-minvar
```

**While the PR is open**: push more commits to the same branch and the PR updates
automatically. Reply to every review comment. Mark a conversation "resolved" only after you have
changed the code.

**When `main` has moved on** (someone else's PR was merged):

```bash
git switch main && git pull
git switch card-a-minvar
git merge main                            # resolve conflicts if any, then:
.venv/bin/python -m pytest tests/teammates -q
git push
```

Conflicts should be rare, because cards do not share files. If you get one in a file you do not
own, stop and ask; do not resolve it by deleting someone else's code.

**Never**:
* `git push --force` to `main`;
* commit data, caches, `.venv/`, `results/` outputs larger than 1 MB, API keys or passwords;
* rewrite history on a branch someone else is also using.

## 4. Running tests and the linter

```bash
# your card's acceptance tests (verbose: shows PASSED / SKIPPED / FAILED per test)
.venv/bin/python -m pytest tests/teammates/test_card_A_minvar.py -v

# one test, or the tests whose name matches a keyword
.venv/bin/python -m pytest tests/teammates/test_card_B_statarb.py -v -k fit_ar1

# the full suite (network and slow tests are excluded by default)
.venv/bin/python -m pytest -q

# tests that download data (needs internet; optional)
.venv/bin/python -m pytest -q -m network

# style / common-bug checks on the files you changed
.venv/bin/ruff check src/kfactors/strategies/minvar.py
```

How to read the results:
* `SKIPPED (Card X not implemented yet ...)` means you have not implemented that function yet.
  This is normal at the start.
* `FAILED` means your implementation does not meet the specification. Read the assertion
  message: it shows the expected and the actual values.
* A card is done when its test file shows **0 failed and 0 skipped**. The only exception is
  Card C's `horns` test, which skips because the optional package cannot be installed.

## 5. Code standards (what reviewers check)

* **Follow the docstring.** The stub's docstring is the specification: inputs, outputs, edge
  cases and errors. Do not change function signatures.
* **numpy-style docstrings** that cite the paper/equation the code implements (already
  provided in the stubs; keep them accurate if you change behaviour).
* **No look-ahead.** A strategy may only use the `window` and `model` it receives. Never reach
  for the full panel or future rows.
* **No silent failure.** Invalid input raises `ValueError`/`TypeError` with a clear message;
  never return NaN weights or swallow exceptions with a bare `except:`.
* **Deterministic.** The same inputs (and seed) give the same outputs. No global random state.
* **No new dependencies** without asking the core team. No `print` in library code.
* **Units**: returns are simple returns in decimals (0.01 = 1%); data matrices are `n x p`
  (rows = dates, columns = assets).

## 6. Using ChatGPT / Claude (encouraged, with rules)

AI assistants are allowed and useful. Each card has suggested prompts. Rules:

1. Paste the **docstring** (the specification) and ask for an implementation of that one
   function. Small, focused requests work best.
2. Make sure you understand every line you commit. Reviewers will ask "why this line?".
3. Never let the assistant edit tests or core files, weaken assertions, or hard-code expected
   values to "make tests pass".
4. Do not paste secrets, tokens or private data into an assistant.
5. If the assistant claims a fact about a paper, check it in the paper before you put it in a
   docstring or the docs.

## 7. Pull request checklist (copy into your PR description)

```
- [ ] Only my card's files changed (`git diff --stat main`)
- [ ] `pytest tests/teammates/<my card test file> -v`: all passed, 0 skipped (paste the summary line)
- [ ] `ruff check <my files>`: All checks passed!
- [ ] Docstrings still match the behaviour
- [ ] No data / cache / large files committed
```

## 8. Getting help

* Read your card in `docs/tasks/`, then `docs/design.md` for the interfaces.
* If you are stuck for more than about 30 minutes, push your branch and open a **draft** PR with
  a question. Showing the failing output is the fastest way to get help.
