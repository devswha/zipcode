# 2026-04-16 — Gemma 4 E4B: "build a game" capability probe

**Model:** `unsloth/gemma-4-E4B-it-GGUF` → `gemma-4-E4B-it-Q4_K_M.gguf`
(SHA256 `dff0ffba4c90b4082d70214d53ce9504a28d4d8d998276dcb3b8881a656c742a`,
4.7 GB, 4096 ctx, all layers offloaded to RTX 2070 SUPER 8 GB).
**Harness:** zipcode @ `35bb45f` (silent-thinking + tool-arg summary build)
running through bundled `llama-server --jinja`.
**Purpose:** measure what a 4B-active local model can actually build
end-to-end inside zipcode's agentic loop, as opposed to the
single-tool probes in `wiki/pages/gemma4-format-spec.md`.

---

## Methodology

Three rounds of strictly increasing difficulty, each in its own empty
directory under `/home/devswha/workspace/test_zipcode/` so the model
can't reach the zipcode source tree:

| Round | Target | Permission | Why this ordering |
|---|---|---|---|
| 1 | Python `guess.py` — single file, stdlib only | `read-only`* | baseline: can it write a correct single file? |
| 2 | HTML5 Canvas Pong — single self-contained `pong.html` | `workspace-write` | adds visual domain, JS/DOM idioms, more lines |
| 3 | Rust `snake_demo` library crate + cargo tests | `full-access` | adds compile loop, test feedback, debug iteration |

\* Round 1 used `read-only` on purpose to observe how zipcode and the
model react when the requested `bash` verification step is blocked.

Each prompt was run with `--ui plain` and `--backend llama-server` so
stdout/stderr/session JSON could be captured as artifacts. Default
silent-thinking mode was used (no `--verbose`) to match real-world UX.

---

## Round 1 — Python guessing game

**Prompt:** write `guess.py` that picks a number 1–100, reads guesses
from stdin, prints too high / too low / correct, and reports attempts;
then verify it starts without crashing.

**Result:** ✅ **Passes cleanly.**

- Tool iterations: **2** (`write_file` + a `bash` call that was blocked
  by `read-only` permission mode, as designed).
- Model correctly recognized that the `bash` step was denied by the
  harness and reported "execution failed due to an environment
  permission denial" in plain language, rather than hallucinating
  success or retrying pointlessly. Good guardrail-awareness.
- Generated code (867 bytes) uses `random.randint(1, 100)`, iterates
  over `sys.stdin`, handles `ValueError` gracefully, prints the three
  required feedback lines, counts attempts, and exits cleanly.
- Manually re-ran it with `printf '50\n25\n12\n6\n3\n1\n2\n'` piped
  in — exit code 0, output matched expectations (`random.randint`
  happened to pick a very low number so all seven guesses printed
  "Too low.", but that's RNG not logic).

**Takeaway:** single-file stdlib Python is squarely inside E4B's
envelope. No surprises, and the permission gate behaved as intended.

---

## Round 2 — HTML5 Canvas Pong

**Prompt:** self-contained `pong.html` with 800×500 canvas, two
paddles (W/S and ↑/↓), bouncing ball, score display, ball reset on
paddle-pass, inline CSS + JS. Read the file back to verify structure.

**Result:** ✅ **Passes cleanly.**

- Tool iterations: **2** (`write_file` + `read_file` self-verification,
  as the prompt requested).
- Output: 220-line `pong.html`, 6034 bytes, single file.
- Structural checks (rg-style):

| Requirement | Found |
|---|---|
| `<canvas id="pongCanvas" width="800" height="500">` | ✓ |
| `getContext('2d')` | ✓ |
| `document.addEventListener('keydown', …)` | ✓ |
| W/S + ArrowUp/ArrowDown branches | ✓ |
| `scoreLeft`, `scoreRight`, `scoreLeft++`, `scoreRight++` | ✓ |
| Score reset / ball re-centering on pass-through | ✓ |
| Inline `<style>` + `<script>` blocks | ✓ |

- **JS syntax sanity:** extracted the `<script>` body to
  `/tmp/pong-extracted.js` and ran `node --check` — **clean**.
- No browser-runtime verification was performed in this probe; static
  analysis is the best we can do headlessly.

**Takeaway:** ~200-line single-file visual app with event handling and
game loop is comfortably within E4B's envelope. The model "knows"
HTML5 Canvas and basic collision idioms. Still a one-shot — no
debugging needed.

---

## Round 3 — Rust `snake_demo` crate with cargo tests

**Prompt:** create `Cargo.toml` + `src/lib.rs` implementing a Snake
`Game` struct with `new`, `tick`, `Direction` and `TickResult` enums,
plus four unit tests. Run `cargo test --offline` and iterate up to
five times if any test fails.

**Result:** ⚠️ **Partial — compiles and 2/5 tests pass, 3 fail, model gave up after one fix cycle.**

### Session transcript (session `78d92a72-4a64-43f2-84ba-5c403d9e855b`)

```
[ 2] model → write_file(Cargo.toml)                       ✓
[ 4] model → write_file(src/lib.rs, 310 lines, 11724 B)   ✓
[ 6] model → bash(cargo test --offline)                   ✗ E0308 mismatched types at src/lib.rs:47
[ 8] model → edit_file(src/lib.rs)                        fixed by wrapping width/2 in a fabricated
                                                          `int_to_i32((width/2).try_into().unwrap())`
                                                          helper it invented on the spot
[10] model → bash(cargo test --offline)                   2 passed, 3 failed (semantic, not compile)
[12] model → (4811 chars of reasoning) → empty content    ← gave up, finish_reason=stop
```

### 3b — resumed session with explicit fix request

After the first prompt exited, I resumed the same session with a
second prompt that told the model exactly which three tests were
failing and what error they expected, and asked it to iterate on each
until all five passed.

```
[13] user   → "Three of the five tests are still failing: … fix one at a time if needed"
[14] model  → (empty content, zero tool_calls) → finish_reason=stop
```

**Zero tool calls on the second attempt.** Not a single `read_file` /
`edit_file` / `bash` invocation. The model acknowledged the failures
internally (long reasoning chain before the silent exit) but declined
to attempt the fix.

### What was actually wrong with the generated code

For the record, the three failing tests were diagnosing real bugs in
`Game::new` / `tick`:

1. **`test_snake_moves_forward_and_tail_follows`** expected a snake of
   length 2 after one tick, but `Game::new` starts it at length 1
   (`VecDeque::from([head])` — the comment claims `[head, tail]` but
   the body is single-element). The tick function presumably doesn't
   grow the snake on non-food moves, so length stays at 1.
2. **`test_eating_food_grows_and_spawns_new_food`** expected `Ate`
   after moving Left into a food placed at `(1, 2)` from a start of
   `(2, 2)` on a 5×5 board, but got `Dead`. Likely root cause:
   collision checks fire before the food-hit check, or the new head
   position is being compared against its own fresh entry in the
   snake deque.
3. **`test_hitting_self_returns_dead`** expected the *first* tick to
   return `Alive`, but got `Dead` — same ordering bug as #2, just
   manifesting even earlier.

These are genuinely subtle but not hard; a human could probably fix
all three in under ten minutes by walking through `tick` with a
debugger. E4B couldn't get from "test output in hand" to "ordering
change in `tick`" on its own.

### Notable flavour: the invented `int_to_i32` helper

On the compile-error retry, E4B did not do the obvious thing
(`let mid_x = width / 2; let mid_y = height / 2;`). Instead it
wrapped each expression in a call to `int_to_i32((width / 2).try_into().unwrap())`
and then defined:

```rust
fn int_to_i32(val: usize) -> i32 {
    val as i32
}
```

This is a pattern-matching "fix" — the error message said "mismatched
types", so the model reached for a type-conversion spell without
understanding that `width` was already `i32` and no conversion was
needed. The code happens to compile because `.try_into().unwrap()`
infers its target from the helper's parameter type. Characteristic
4B-active behavior: it knows the tokens `try_into`, `unwrap`, `as i32`,
and fuses them into something plausible-looking that works
mechanically but misses the underlying model.

---

## Cross-round observations

1. **Success ceiling is around "one file, one concept, no debug
   loop".** Rounds 1 and 2 fit that description and were one-shot
   successes. Round 3 adds a real feedback loop and that's where E4B
   starts to plateau.

2. **The debug loop does work, for one hop.** E4B successfully read
   the compile error in Round 3, localized the line, and edited the
   file. It recovered from a compile failure without human help.
   That's non-trivial and it's what differentiates a 4B-active
   agentic model from a pure completion model.

3. **The second debug hop is where it gives up.** Both in the
   original Round 3 run and in the 3b resume, as soon as the failure
   was *semantic* (test assertion, not compile error), the model
   produced a long internal reasoning chain and then exited with zero
   tool calls. Not a loop cap, not a context overflow, not a format
   error — just a learned "this isn't going anywhere" judgment.
   Healthy behavior in that it doesn't waste tool calls on a dead end,
   painful in that it declines even a single exploratory
   `read_file` + `edit_file` attempt.

4. **Silent-thinking mode worked as designed across all three
   rounds.** No reasoning leaked into session history (verified
   against the JSON), `thinking (N chars)` counter ticked up into the
   thousands during hard turns without dumping anything to stdout,
   and tool-call argument summaries kept 12 KB `write_file` bodies
   from scrolling the terminal. This is exactly what Phase A'/A'''
   were supposed to buy and it held up under real dogfood load.

5. **Permission gates stay sane under agentic load.** Round 1's
   blocked `bash` call was reported back through the tool output as a
   denial and the model integrated that into its final message
   without hallucinating success. Round 3 ran under `full-access`
   deliberately so the model could run `cargo test`, and it did.

6. **Tool-selection quality on the 10-tool registry is acceptable but
   not stellar.** The model picks the right tool on explicit asks
   (`write_file`, `bash`, `edit_file`) but it never reached for
   `tool_search` or `grep_search` to explore its own generated code
   before attempting a fix. The `Phase B` wiki observation from the
   previous probe still holds: on exploratory queries, E4B doesn't
   instinctively reach for `grep_search`.

---

## What this implies for zipcode's product envelope

If we're honest about where E4B lives today, the reasonable pitch is:

- **"Write me X"** where X is a self-contained file and the user
  knows what they want → ✅ reliable.
- **"Explain / navigate this repo"** → ✅ as long as the exploration
  tool set is the right choice, which it often isn't by default.
- **"Fix this bug for me"** where "the bug" is a compiler error →
  ⚠️ one shot often works; two shots rarely.
- **"Build this feature end-to-end with tests"** → ❌ not yet.
  Needs a bigger model (26B A4B or 31B) or a harness that splits
  the work into planner → implement → test → fix sub-agents so that
  each sub-agent is doing one-hop work.

This isn't a bug in zipcode — the wire contract, thinking plumbing,
permission system, and tool loop all behaved correctly under real
dogfood. It's the realistic ceiling of a 4B-active local model, and
it's why the `wiki/pages/gemma4-format-spec.md` "Phase B" section
already flagged "prompt-engineering" and "bigger model" as the two
axes of improvement, not "fix the template".

---

## Artifacts

Scratch paths on the probe host (not committed):

- `/home/devswha/workspace/test_zipcode/round1/guess.py` (867 B)
- `/home/devswha/workspace/test_zipcode/round2/pong.html` (6034 B)
- `/home/devswha/workspace/test_zipcode/round3/` — Cargo.toml,
  src/lib.rs (310 lines), target/ (cargo test cache), Cargo.lock
- `~/.zipcode/sessions/78d92a72-4a64-43f2-84ba-5c403d9e855b.json`
  (15 messages total across both Round 3 prompts)
- `/tmp/round{1,2,3,3b}.{stdout,stderr}` — raw zipcode output streams

Regenerate by running the commands in the "Prompt" blocks of each
round above against the same GGUF. E4B is mostly deterministic at
temperature 0.2 (zipcode's default), so outcomes should be stable
modulo the `random.randint` in Round 1.

---

## Next probes worth doing

1. **Re-run Round 3 with `--verbose`** so the reasoning channel is
   visible and we can read what the model was actually thinking
   during the second "give up" turn. Might reveal a concrete
   blocker (e.g. "I can't see the current lib.rs content any more")
   vs a generic uncertainty.
2. **Re-run Round 3 with an explicit `read_file` nudge in the
   prompt** ("Start by reading src/lib.rs, then describe the current
   tick implementation, then decide what to change") to see if
   forcing the exploration step breaks the silent-exit pattern.
3. **Re-run Round 3 against the 26B A4B build** (Jiunsong's
   `supergemma4-26b-uncensored-gguf-v2`) once 24 GB VRAM is
   available, and compare second-hop debug behavior. This is the
   most direct apples-to-apples measurement of the "bigger model
   fixes this" hypothesis.
4. **Add a `BASE_SYSTEM_PROMPT` directive** that explicitly tells
   the model "when tests fail, re-read the file before editing, and
   do not stop until all tests pass or you have tried at least three
   distinct fixes". See if that changes Round 3b behavior without
   touching model weights.
