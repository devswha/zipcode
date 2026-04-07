---
title: Test Scenarios
tags: [troubleshooting]
sources: [session-2026-04-08]
updated: 2026-04-08
---

# Test Scenarios

Validated test scenarios for zipcode functionality.

## Basic Scenarios (2026-04-08, all passing)

### Test 1: Basic Conversation
- **Prompt**: "What is 2+2? Just give me the number."
- **Expected**: Numeric answer
- **Result**: ✅ "4"

### Test 2: File Read with Metadata
- **Prompt**: "Read the file main.rs and tell me what it does."
- **Expected**: read_file tool call, metadata header in output
- **Result**: ✅ Tool called, `[file: path, lines: 1-1 of 1 total]` header shown, content explained

### Test 3: Binary File Rejection
- **Prompt**: "Read the file binary.dat"
- **Expected**: Error with "file appears to be binary"
- **Result**: ✅ Error returned with absolute path, model explained the limitation

### Test 4: Grep Search
- **Prompt**: "Search for TODO comments in all files"
- **Expected**: grep_search tool call finding TODO in main.rs
- **Result**: ✅ Found `main.rs:1: fn main() { ... // TODO: add error handling }`

### Test 5: Bash Execution (full-access)
- **Prompt**: "Run ls -la and tell me what files are here" (--permission-mode full-access)
- **Expected**: bash tool call, file listing
- **Result**: ✅ `ls -la` executed, files listed and summarized

### Test 6: Read-Only Permission
- **Prompt**: "Create a file called test.txt" (--permission-mode read-only)
- **Expected**: Refusal without tool call
- **Result**: ✅ Model refused without attempting write_file

## Test Environment

- Model: Gemma 4 E2B IT Q8_0 (4.6GB)
- Backend: llama-server with GPU offload (RTX 2070 SUPER)
- Flags: ZIPCODE_GPU_LAYERS=99, ZIPCODE_FLASH_ATTENTION=1

## Advanced Scenarios (2026-04-08, all passing, 2 behavioral issues)

### Test 7: Multi-tool Chain (grep + read + summarize)
- **Prompt**: "Find all TODO and FIXME comments across all .rs files"
- **Expected**: grep_search + read_file chain, structured summary
- **Result**: ✅ Found 7 comments across 3 files, categorized by priority (FIXME first, then TODO)

### Test 8: Edit File (targeted replacement)
- **Prompt**: "Replace the unwrap() call in load_config with proper error handling using match"
- **Expected**: edit_file tool call with targeted replacement
- **Result**: ⚠️ Used write_file (full overwrite) instead of edit_file. Code correct but approach risky for large files
- **Issue**: Model prefers write_file over edit_file. See [Model Behavioral Issues](#model-behavioral-issues)

### Test 9: Write New File
- **Prompt**: "Create a new file src/utils/mod.rs with pub mod helpers;"
- **Expected**: write_file tool call
- **Result**: ✅ File created correctly

### Test 10: Bash + File Analysis
- **Prompt**: "Run wc -l on all .rs files and tell me which has the most lines"
- **Expected**: bash tool with recursive file discovery
- **Result**: ⚠️ glob_search used `*.rs` (non-recursive) instead of `**/*.rs`, missed src/ files
- **Issue**: Model used wrong glob pattern. See [Model Behavioral Issues](#model-behavioral-issues)

### Test 11: Glob Search (recursive)
- **Prompt**: "Find all Rust files in this project using glob search"
- **Expected**: glob_search with `**/*.rs`
- **Result**: ✅ Found all 4 .rs files

### Test 12: Nonexistent File
- **Prompt**: "Read the file src/lib.rs"
- **Expected**: Error with absolute path
- **Result**: ✅ "failed to stat file: /absolute/path/src/lib.rs", model explained clearly

## Model Behavioral Issues

These are not tool bugs but model tendencies that can be improved via system prompt:

### Issue 1: write_file over edit_file preference
- **Observed**: When asked to modify part of a file, model uses write_file (full overwrite) instead of edit_file (targeted replacement)
- **Risk**: Large files could lose content if model makes mistakes in rewriting
- **Mitigation**: System prompt guidance to prefer edit_file for modifications

### Issue 2: Non-recursive glob patterns
- **Observed**: Model sometimes uses `*.rs` instead of `**/*.rs`, missing subdirectory files
- **Risk**: Incomplete search results
- **Mitigation**: System prompt guidance to use `**/*` for project-wide searches

## See Also
- [[tool-system]]
- [[inference-backends]]
- [[runtime-loop]]
