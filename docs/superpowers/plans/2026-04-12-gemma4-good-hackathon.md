# Gemma 4 Good Hackathon Execution Plan

**Competition:** [The Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
**Deadline:** 2026-05-18 23:59 UTC (~5 weeks)
**Prize Pool:** $200,000
**Track:** Education
**Pitch:** Offline AI Coding Tutor for Underserved Communities

---

## Strategy

### Positioning

> zipcode: a local-only AI coding agent that runs Gemma 4 offline by default,
> enabling coding education in environments with no internet access.

Target users: students in developing regions, rural schools, military/restricted facilities, disaster relief zones.

Key differentiators vs other submissions:
1. **Rust single binary** — zero install, zero runtime dependencies
2. **Offline by default** — no cloud calls; explicit user-requested GitHub repository fetching is the network exception. USB stick deployment.
3. **Agentic, not chatbot** — file editing, code execution, search. Students learn by doing.
4. **Gemma 4 edge models** — E2B/E4B run on consumer hardware without GPU

### Domain Framing (Education)

Reframe existing tools as educational features:

| Current Tool | Education Framing |
|---|---|
| FileEdit | Student code review & guided fixes |
| BashExec | Code execution with safety sandbox |
| GrepSearch | "Find where this pattern is used" exercises |
| FileRead | Reading and understanding existing codebases |
| ListDir / GlobSearch | Project structure exploration |
| TodoWrite | Learning task tracking |

### Competition Requirements Checklist

- [ ] Working demo (CLI REPL with tutor mode)
- [ ] Public GitHub repository
- [ ] Technical write-up (how Gemma 4 is applied)
- [ ] Short demo video (~3 min)
- [ ] Kaggle submission

---

## Phase 1: Foundation (Apr 12 - Apr 20)

### Task 1.1: Gemma 4 Model Validation

Verify Gemma 4 GGUF models work end-to-end with llama-server backend.

- [ ] Download Gemma 4 E4B GGUF (quantized, ~3GB)
- [ ] Download Gemma 4 12B GGUF (quantized, ~8GB) as secondary option
- [ ] Test tool-calling with `<tool_call>` tags — confirm Gemma 4 follows the format
- [ ] Benchmark: tokens/sec on consumer hardware (no GPU, 16GB RAM)
- [ ] Document which quant levels (Q4_K_M, Q5_K_M, Q8_0) are usable

**Risk:** Gemma 4 may use a different tool-call format than Gemma 2/3.
**Mitigation:** Check Gemma 4 docs for native function calling format, adapt chat template if needed.

### Task 1.2: Chat Template Update for Gemma 4

Gemma 4 has native function calling support — may differ from current hardcoded template.

- [ ] Research Gemma 4's official function calling / tool use format
- [ ] Update `crates/inference/src/chat_template.rs` if format changed
- [ ] Update system prompt to include tool definitions in Gemma 4's expected schema
- [ ] Test: multi-turn conversation with tool calls works correctly

**Files:**
- `crates/inference/src/chat_template.rs`
- `crates/runtime/src/conversation.rs` (system prompt construction)

### Task 1.3: Tutor Mode System Prompt

Add an education-focused system prompt mode.

- [ ] Create tutor persona: patient, encouraging, Socratic method
- [ ] Include pedagogical instructions: explain step-by-step, ask guiding questions before giving answers, encourage experimentation
- [ ] Support multiple skill levels (beginner, intermediate, advanced)
- [ ] Add `--tutor` CLI flag to activate tutor mode
- [ ] Add `tutor_mode` config option in `.zipcode.json`

**Files:**
- `crates/runtime/src/system_prompt.rs` (new or extend existing)
- `crates/runtime/src/config.rs` (add tutor config fields)
- `crates/cli/src/main.rs` (add --tutor flag)

---

## Phase 2: Education Features (Apr 21 - May 4)

### Task 2.1: Guided Exercise Framework

Simple exercise system that the tutor can reference.

- [ ] Define exercise format: markdown files with problem statement, hints, solution
- [ ] Create `.zipcode-exercises/` directory convention
- [ ] Add `ExerciseLoad` tool or extend FileRead to recognize exercise files
- [ ] Ship 5-10 sample exercises (Python basics, file I/O, simple algorithms)
- [ ] Tutor prompt knows how to load and walk through exercises

### Task 2.2: Beginner-Friendly Error Explanations

When BashExec returns errors, the tutor should explain them accessibly.

- [ ] Add error-explanation prompt injection: "explain this error to a beginner"
- [ ] Common error patterns: syntax errors, import failures, type mismatches
- [ ] Localization hooks for non-English error messages (future)

### Task 2.3: Progress Tracking

Simple local progress tracking for students.

- [ ] Extend TodoWrite or create lightweight progress tracker
- [ ] Track: exercises attempted, completed, topics covered
- [ ] Store at `~/.zipcode/progress.json`
- [ ] Tutor can reference progress: "You've completed 3/10 exercises"

### Task 2.4: Safety Hardening for Education

Students will run arbitrary code — tighten the sandbox.

- [ ] Review BashExec timeout and resource limits
- [ ] Add `--safe-mode` flag: blocks destructive commands (rm -rf, sudo, etc.)
- [ ] Ensure path validation prevents escaping workspace
- [ ] Document safety model in write-up

---

## Phase 3: Polish & Submission (May 5 - May 18)

### Task 3.1: Packaging & Deployment

Make the USB-stick story bulletproof.

- [ ] Update `scripts/package.sh` to bundle Gemma 4 GGUF + binary + sample exercises
- [ ] Test on fresh Ubuntu 22.04 (no dev tools installed)
- [ ] Test on a machine with no internet access
- [ ] Document hardware requirements (minimum: 8GB RAM, recommended: 16GB)
- [ ] Create one-line install: `./install.sh` from USB

### Task 3.2: Technical Write-Up

Required for submission.

- [ ] Problem statement: coding education gap in offline environments
- [ ] Solution: zipcode + Gemma 4 as offline AI coding tutor
- [ ] Architecture diagram (existing, polish)
- [ ] How Gemma 4 is used: model selection rationale, function calling, edge deployment
- [ ] Impact section: who benefits, how, scalability
- [ ] Limitations and future work
- [ ] Save as `docs/hackathon-writeup.md`

### Task 3.3: Demo Video (~3 min)

- [ ] Script the demo flow:
  1. Show USB stick / single binary
  2. Launch zipcode in tutor mode (no internet)
  3. Student asks "teach me Python basics"
  4. Tutor creates a file, explains, asks student to modify
  5. Student makes an error, tutor explains and guides
  6. Show progress tracking
- [ ] Record with terminal recorder (asciinema or OBS)
- [ ] Add minimal narration or captions

### Task 3.4: Public Repository

- [ ] Clean up repo for public visibility
- [ ] Write hackathon-focused README section
- [ ] Ensure no secrets, no large binaries committed
- [ ] Add Apache 2.0 or MIT license if not present
- [ ] Tag release: `v0.1.0-hackathon`

### Task 3.5: Kaggle Submission

- [ ] Register team on Kaggle
- [ ] Submit per competition format
- [ ] Link to GitHub repo, write-up, video

---

## Timeline Summary

```
Week 1 (Apr 12-18): Gemma 4 validation, chat template, tutor prompt
Week 2 (Apr 19-25): Exercise framework, error explanations
Week 3 (Apr 26-May 2): Progress tracking, safety hardening
Week 4 (May 3-9):  Packaging, write-up draft
Week 5 (May 10-18): Demo video, polish, submit
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gemma 4 tool-call format incompatible | Medium | High | Research early (Task 1.2), adapt template |
| Gemma 4 E4B too slow on CPU | Medium | Medium | Fall back to E2B, or recommend 12B with GPU |
| llama-server binary not portable | Low | High | Static build or bundle prebuilt binary |
| Scope creep on education features | High | Medium | Stick to this plan, cut Phase 2 items if behind |
| Demo video quality | Low | Medium | Use asciinema for clean terminal recording |

---

## What NOT To Build

- Web UI (terminal-only is fine, and more authentic for coding education)
- Multi-language model support (Gemma 4 only)
- Cloud deployment (defeats the point)
- Complex curriculum management (keep it simple)
- User authentication (single user, local only)

---

## Success Criteria

1. zipcode runs Gemma 4 E4B offline on a 16GB RAM laptop
2. Tutor mode guides a complete beginner through a Python exercise
3. Everything fits on a USB stick (<8GB total)
4. Demo video clearly shows the offline education story
5. Technical write-up explains Gemma 4 usage convincingly
