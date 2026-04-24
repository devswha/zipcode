use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SkillParameter {
    pub name: String,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SkillFrontmatter {
    name: String,
    description: String,
    #[serde(default)]
    tool_allowlist: Vec<String>,
    #[serde(default)]
    parameters: Vec<SkillParameter>,
}

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub tool_allowlist: Vec<String>,
    pub parameters: Vec<SkillParameter>,
    pub body: String,
}

impl Skill {
    pub fn parse(src: &str) -> Result<Self> {
        // Line-based parser: file must start with `---`, frontmatter ends at the
        // next line that equals `---`. Body may freely contain `---` separators.
        let mut lines = src.split_inclusive('\n');
        let first = lines.next().unwrap_or("");
        if first.trim_end_matches(['\r', '\n']) != "---" {
            bail!("missing frontmatter: content does not start with '---'");
        }

        let mut yaml_buf = String::new();
        let mut body_buf = String::new();
        let mut closed = false;

        for line in lines {
            if !closed && line.trim_end_matches(['\r', '\n']) == "---" {
                closed = true;
                continue;
            }
            if closed {
                body_buf.push_str(line);
            } else {
                yaml_buf.push_str(line);
            }
        }

        if !closed {
            bail!("missing frontmatter closing '---'");
        }

        let fm: SkillFrontmatter = serde_yaml::from_str(&yaml_buf)
            .with_context(|| format!("malformed YAML frontmatter: {yaml_buf}"))?;

        let body = body_buf.trim_start_matches('\n').to_owned();

        Ok(Self {
            name: fm.name,
            description: fm.description,
            tool_allowlist: fm.tool_allowlist,
            parameters: fm.parameters,
            body,
        })
    }

    pub fn render(&self, params: &HashMap<String, String>) -> String {
        let defaults: HashMap<&str, &str> = self
            .parameters
            .iter()
            .filter_map(|p| p.default.as_deref().map(|d| (p.name.as_str(), d)))
            .collect();

        let mut result = self.body.clone();
        let mut output = String::with_capacity(result.len());

        loop {
            match result.find("{{") {
                None => {
                    output.push_str(&result);
                    break;
                }
                Some(start) => {
                    output.push_str(&result[..start]);
                    match result[start..].find("}}") {
                        None => {
                            output.push_str(&result[start..]);
                            break;
                        }
                        Some(rel_end) => {
                            let end = start + rel_end + 2;
                            let key = result[start + 2..start + rel_end].trim();
                            let value = params
                                .get(key)
                                .map(String::as_str)
                                .or_else(|| defaults.get(key).copied())
                                .unwrap_or("");
                            output.push_str(value);
                            result = result[end..].to_owned();
                        }
                    }
                }
            }
        }

        output
    }
}

pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    pub fn load_from(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            return Ok(Self::new());
        }

        let mut registry = Self::new();

        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read skills directory: {}", dir.display()))?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("failed to read directory entry: {e}");
                    continue;
                }
            };

            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }

            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("failed to read skill file {}: {e}", path.display());
                    continue;
                }
            };

            let skill = match Skill::parse(&content) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("skipping malformed skill file {}: {e}", path.display());
                    continue;
                }
            };

            if registry.skills.contains_key(&skill.name) {
                tracing::warn!(
                    "duplicate skill name '{}' in {}; keeping first",
                    skill.name,
                    path.display()
                );
                continue;
            }

            registry.skills.insert(skill.name.clone(), skill);
        }

        Ok(registry)
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.skills.keys().map(String::as_str).collect()
    }

    pub fn catalog_for_prompt(&self) -> String {
        const MAX_CHARS: usize = 2000;

        let mut lines: Vec<String> = self
            .skills
            .values()
            .map(|s| format!("{}: {}", s.name, s.description))
            .collect();
        lines.sort();

        let joined = lines.join("\n");
        if joined.len() <= MAX_CHARS {
            joined
        } else {
            // MAX_CHARS is a byte budget; find the largest safe char boundary
            // at or below it so multi-byte UTF-8 (Korean/CJK/emoji) cannot panic.
            let mut end = MAX_CHARS;
            while end > 0 && !joined.is_char_boundary(end) {
                end -= 1;
            }
            joined[..end].to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn sample_skill_str() -> &'static str {
        "---\nname: code-review\ndescription: Review recently changed code\ntool_allowlist:\n  - read_file\n  - grep_search\nparameters:\n  - name: scope\n    default: HEAD~1..HEAD\n---\nYou are a code reviewer. Focus on changes in {{ scope }}.\n"
    }

    #[test]
    fn test_skill_from_str_parses_frontmatter() {
        let skill = Skill::parse(sample_skill_str()).unwrap();
        assert_eq!(skill.name, "code-review");
        assert_eq!(skill.description, "Review recently changed code");
        assert_eq!(skill.tool_allowlist, vec!["read_file", "grep_search"]);
        assert_eq!(skill.parameters.len(), 1);
        assert_eq!(skill.parameters[0].name, "scope");
        assert_eq!(skill.parameters[0].default.as_deref(), Some("HEAD~1..HEAD"));
        assert!(skill.body.contains("code reviewer"));
    }

    #[test]
    fn test_skill_from_str_rejects_missing_frontmatter() {
        let err = Skill::parse("no frontmatter here").unwrap_err();
        assert!(err.to_string().contains("missing frontmatter"));
    }

    #[test]
    fn test_skill_from_str_rejects_malformed_yaml() {
        let src = "---\nname: [unclosed bracket\ndescription: bad\n---\nbody\n";
        let err = Skill::parse(src).unwrap_err();
        assert!(err.to_string().contains("malformed YAML"));
    }

    #[test]
    fn test_skill_render_substitutes_params() {
        let skill = Skill::parse(sample_skill_str()).unwrap();
        let mut params = HashMap::new();
        params.insert("scope".to_owned(), "main..feature".to_owned());
        let rendered = skill.render(&params);
        assert!(rendered.contains("main..feature"));
        assert!(!rendered.contains("{{ scope }}"));
    }

    #[test]
    fn test_skill_render_uses_default_when_missing() {
        let skill = Skill::parse(sample_skill_str()).unwrap();
        let params = HashMap::new();
        let rendered = skill.render(&params);
        assert!(rendered.contains("HEAD~1..HEAD"));
    }

    #[test]
    fn test_skill_render_empty_when_no_default_no_value() {
        let src =
            "---\nname: s\ndescription: d\nparameters:\n  - name: foo\n---\nHello {{ foo }}!\n";
        let skill = Skill::parse(src).unwrap();
        let params = HashMap::new();
        let rendered = skill.render(&params);
        assert_eq!(rendered.trim_end(), "Hello !");
    }

    #[test]
    fn test_skill_render_ignores_unknown_placeholders() {
        let src = "---\nname: s\ndescription: d\n---\nHello {{ unknown }}!\n";
        let skill = Skill::parse(src).unwrap();
        let params = HashMap::new();
        let rendered = skill.render(&params);
        assert_eq!(rendered.trim_end(), "Hello !");
    }

    #[test]
    fn test_skill_registry_load_from_missing_dir_empty() {
        let registry = SkillRegistry::load_from(Path::new("/nonexistent/path/xyz")).unwrap();
        assert!(registry.names().is_empty());
    }

    fn write_skill_file(dir: &TempDir, filename: &str, content: &str) {
        let path = dir.path().join(filename);
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn test_skill_registry_load_from_walks_md_files() {
        let dir = TempDir::new().unwrap();
        write_skill_file(
            &dir,
            "review.md",
            "---\nname: review\ndescription: Code review\n---\nReview body\n",
        );
        write_skill_file(
            &dir,
            "search.md",
            "---\nname: search\ndescription: Search code\n---\nSearch body\n",
        );
        write_skill_file(&dir, "notes.txt", "not a markdown file");

        let registry = SkillRegistry::load_from(dir.path()).unwrap();
        let mut names = registry.names();
        names.sort();
        assert_eq!(names, vec!["review", "search"]);
    }

    #[test]
    fn test_skill_registry_load_from_skips_malformed_file() {
        let dir = TempDir::new().unwrap();
        write_skill_file(&dir, "bad.md", "no frontmatter at all");
        write_skill_file(
            &dir,
            "good.md",
            "---\nname: good\ndescription: Good skill\n---\nBody\n",
        );

        let registry = SkillRegistry::load_from(dir.path()).unwrap();
        assert!(registry.get("good").is_some());
        assert!(registry.get("bad").is_none());
        assert_eq!(registry.names().len(), 1);
    }

    #[test]
    fn test_skill_registry_get_returns_skill() {
        let dir = TempDir::new().unwrap();
        write_skill_file(
            &dir,
            "myskill.md",
            "---\nname: myskill\ndescription: My skill\n---\nBody\n",
        );
        let registry = SkillRegistry::load_from(dir.path()).unwrap();
        let skill = registry.get("myskill").unwrap();
        assert_eq!(skill.name, "myskill");
    }

    #[test]
    fn test_skill_registry_catalog_for_prompt_concats_name_description() {
        let dir = TempDir::new().unwrap();
        write_skill_file(
            &dir,
            "a.md",
            "---\nname: alpha\ndescription: Alpha skill\n---\nBody\n",
        );
        write_skill_file(
            &dir,
            "b.md",
            "---\nname: beta\ndescription: Beta skill\n---\nBody\n",
        );
        let registry = SkillRegistry::load_from(dir.path()).unwrap();
        let catalog = registry.catalog_for_prompt();
        assert!(catalog.contains("alpha: Alpha skill"));
        assert!(catalog.contains("beta: Beta skill"));
    }

    #[test]
    fn test_skill_registry_catalog_for_prompt_truncates_over_2000_chars() {
        let dir = TempDir::new().unwrap();
        let long_desc = "x".repeat(2100);
        let content = format!("---\nname: longskill\ndescription: {long_desc}\n---\nBody\n");
        write_skill_file(&dir, "long.md", &content);
        let registry = SkillRegistry::load_from(dir.path()).unwrap();
        let catalog = registry.catalog_for_prompt();
        assert!(catalog.len() <= 2000);
    }

    #[test]
    fn test_skill_registry_catalog_truncates_on_utf8_char_boundary() {
        let dir = TempDir::new().unwrap();
        // Korean 3-byte chars; 700 * 3 = 2100 bytes; the 2000-byte cut lands
        // mid-codepoint. Naive slicing would panic.
        let long_desc = "가".repeat(700);
        let content = format!("---\nname: kr\ndescription: {long_desc}\n---\nBody\n");
        write_skill_file(&dir, "kr.md", &content);
        let registry = SkillRegistry::load_from(dir.path()).unwrap();
        let catalog = registry.catalog_for_prompt(); // must not panic
        assert!(catalog.len() <= 2000);
        assert!(catalog.is_char_boundary(catalog.len()));
    }

    #[test]
    fn test_skill_from_str_preserves_body_containing_triple_dash() {
        let src = "---\nname: doc\ndescription: d\n---\nSection one.\n\n---\n\nSection two separated by horizontal rule.\n";
        let skill = Skill::parse(src).unwrap();
        assert_eq!(skill.name, "doc");
        assert!(skill.body.contains("Section one."));
        assert!(skill.body.contains("---"));
        assert!(skill.body.contains("Section two"));
    }

    #[test]
    fn test_skill_from_str_accepts_crlf_frontmatter() {
        let src = "---\r\nname: s\r\ndescription: d\r\n---\r\nBody line\r\n";
        let skill = Skill::parse(src).unwrap();
        assert_eq!(skill.name, "s");
        assert!(skill.body.contains("Body line"));
    }
}
