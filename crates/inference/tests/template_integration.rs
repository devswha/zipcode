//! Integration tests — model-aware chat template selection and provider wiring.
//!
//! Tests 1–5: [`GemmaTemplate`] via the [`ChatTemplate`] trait object.
//! Test  6:   [`TemplateRegistry`] selects the correct template for a model path.
//! Test  7:   [`MockInferenceProvider::with_template`] builder wires through correctly.
//!
//! ## Registry isolation note
//!
//! Once worker-2 adds `~/.zipcode/models/registry.json` override support, any
//! test that exercises JSON loading must use `tempfile::tempdir()` to redirect
//! the registry file path so it does not read from or write to the real path.
//! Tests below that use only in-memory / filename-based selection are safe
//! without `tempfile` isolation.

use zipcode_inference::chat_template::ToolSpec;
use zipcode_inference::types::ChatMessage;
use zipcode_inference::{ChatTemplate, GemmaTemplate, MockInferenceProvider, MockResponse};

// ── Test 1 ──────────────────────────────────────────────────────────────────

/// `GemmaTemplate` must identify itself as `"gemma_native"`.
/// The name is used by the runtime and registry to match templates to backends.
#[test]
fn gemma_template_name_returns_gemma_native() {
    let t = GemmaTemplate;
    assert_eq!(t.name(), "gemma_native");
}

// ── Test 2 ──────────────────────────────────────────────────────────────────

/// `GemmaTemplate` uses structured JSON tool calls, so `native_tool_calling`
/// must be `true`. An emulator template (worker-2) would return `false`.
#[test]
fn gemma_template_native_tool_calling_is_true() {
    let t = GemmaTemplate;
    assert!(
        t.native_tool_calling(),
        "GemmaTemplate must report native tool-calling support"
    );
}

// ── Test 3 ──────────────────────────────────────────────────────────────────

/// `parse_tool_calls` works correctly when dispatched through a trait object.
/// This proves the vtable is wired up and the method behaves identically to
/// calling it on the concrete type.
#[test]
fn gemma_template_parse_tool_calls_via_trait_object() {
    let t: Box<dyn ChatTemplate> = Box::new(GemmaTemplate);

    let output = concat!(
        "<tool_call>\n",
        "{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"src/main.rs\"}}\n",
        "</tool_call>"
    );
    let calls = t.parse_tool_calls(output);

    assert_eq!(calls.len(), 1, "expected exactly one tool call");
    assert_eq!(calls[0].name, "read_file");
    assert_eq!(calls[0].arguments["file_path"], "src/main.rs");
}

// ── Test 4 ──────────────────────────────────────────────────────────────────

/// `extract_text_content` strips tool-call blocks and leaves only plain text,
/// dispatched via a trait object so the vtable path is exercised end-to-end.
#[test]
fn gemma_template_extract_text_via_trait_object() {
    let t: Box<dyn ChatTemplate> = Box::new(GemmaTemplate);

    let output = "Hello <tool_call>{\"name\":\"bash\",\"arguments\":{}}</tool_call> world";
    let text = t.extract_text_content(output);

    assert_eq!(text, "Hello  world");
    assert!(!text.contains("<tool_call>"));
}

// ── Test 5 ──────────────────────────────────────────────────────────────────

/// `render_prompt` injects the system message as a prefixed turn and ends with
/// the model primer.  Verified via a trait object so the signature contract is
/// enforced.
#[test]
fn gemma_template_render_prompt_with_system_injects_system_turn() {
    let t: Box<dyn ChatTemplate> = Box::new(GemmaTemplate);

    let messages = vec![ChatMessage::user("hello")];
    let tools: Vec<ToolSpec> = vec![];
    let prompt = t.render_prompt(&messages, "you are helpful", &tools);

    assert!(
        prompt.contains("you are helpful"),
        "system prompt must appear in rendered output"
    );
    assert!(
        prompt.contains("<start_of_turn>user"),
        "user turn marker must be present"
    );
    assert!(
        prompt.ends_with("<start_of_turn>model\n"),
        "prompt must end with the model generation primer"
    );
}

// ── Test 6 ──────────────────────────────────────────────────────────────────

/// [`zipcode_inference::default_registry`] selects the correct template for a
/// model filename via glob pattern matching.
///
/// - Gemma 4 GGUF → `"gemma_native"`
/// - Qwen 2.5 GGUF → `"chatml"`
/// - Llama 3.1 GGUF → `"llama31_json"`
#[test]
fn template_registry_selects_correct_template_for_model_filename() {
    use zipcode_inference::default_registry;

    let registry = default_registry();

    // Gemma 4
    let template = registry.template_for_model("gemma-4-e4b-Q4_K_M.gguf");
    assert_eq!(
        template.name(),
        "gemma_native",
        "registry should return GemmaTemplate for a Gemma GGUF filename"
    );

    // Qwen 2.5
    let template = registry.template_for_model("qwen2.5-0.5b-instruct.gguf");
    assert_eq!(
        template.name(),
        "chatml",
        "registry should return ChatMLTemplate for a Qwen2.5 filename"
    );

    // Llama 3.1
    let template = registry.template_for_model("llama-3.1-8b-instruct.Q4_K_M.gguf");
    assert_eq!(
        template.name(),
        "llama31_json",
        "registry should return Llama31Template for a Llama 3.1 filename"
    );
}

// ── Test 7 ──────────────────────────────────────────────────────────────────

/// [`MockInferenceProvider::with_template`] stores the supplied template and
/// makes it accessible via [`MockInferenceProvider::template`].  This proves
/// the builder path compiles and the stored value is the one passed in.
#[test]
fn mock_provider_with_template_builder_stores_template() {
    let mock = MockInferenceProvider::new(vec![MockResponse::Text("ok".to_string())])
        .with_template(Box::new(GemmaTemplate));

    assert_eq!(
        mock.template().name(),
        "gemma_native",
        "with_template() must store the supplied template and expose it via template()"
    );
    assert!(
        mock.template().native_tool_calling(),
        "the stored template should report native_tool_calling correctly"
    );
}
