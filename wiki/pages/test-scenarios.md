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

## See Also
- [[tool-system]]
- [[inference-backends]]
- [[runtime-loop]]
