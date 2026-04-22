use unicode_width::UnicodeWidthChar;

/// Total display width of a string in terminal columns.
/// CJK chars count as 2, control chars as 0, tab as 1.
#[allow(dead_code)]
pub fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Truncate a string to fit within `max_width` display columns.
/// Appends "…" if truncated. Returns empty string if `max_width` < 2.
pub fn truncate_display(s: &str, max_width: usize) -> String {
    if max_width < 2 {
        return String::new();
    }
    let mut width = 0usize;
    let mut end = 0usize;
    for (i, ch) in s.char_indices() {
        let w = char_width(ch);
        if width + w > max_width {
            break;
        }
        width += w;
        end = i + ch.len_utf8();
    }
    if end == s.len() {
        s.to_string()
    } else {
        // We need room for "…" (1 column). Shrink if needed.
        let mut truncated_width = 0usize;
        let mut truncated_end = 0usize;
        for (i, ch) in s.char_indices() {
            let w = char_width(ch);
            if truncated_width + w > max_width.saturating_sub(1) {
                break;
            }
            truncated_width += w;
            truncated_end = i + ch.len_utf8();
        }
        format!("{}…", &s[..truncated_end])
    }
}

/// Wrap a string at `max_width` display columns. Does NOT handle newlines —
/// caller should split on '\n' first. Returns at least one element.
pub fn wrap_display(s: &str, max_width: usize) -> Vec<String> {
    let max_width = max_width.max(1);
    if s.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for ch in s.chars() {
        let w = char_width(ch);
        if current_width + w > max_width && !current.is_empty() {
            out.push(current);
            current = String::new();
            current_width = 0;
        }
        current.push(ch);
        current_width += w;
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Display column at a given byte cursor position within a string.
pub fn column_at_byte(s: &str, byte_cursor: usize) -> usize {
    let cursor = floor_char_boundary(s, byte_cursor);
    s[..cursor].chars().map(char_width).sum()
}

/// Byte offset corresponding to a target display column.
/// Returns the byte offset at or just before the target column.
pub fn byte_at_column(s: &str, target_col: usize) -> usize {
    let mut col = 0usize;
    for (i, ch) in s.char_indices() {
        if col >= target_col {
            return i;
        }
        let width = char_width(ch);
        if target_col < col + width {
            return i;
        }
        col += width;
    }
    s.len()
}

/// Display width of a single character.  Exposed as `pub(crate)` so that
/// `tui_composer` can compute per-character widths without pulling in
/// `unicode_width` directly.
pub fn char_display_width(ch: char) -> usize {
    char_width(ch)
}

fn char_width(ch: char) -> usize {
    if ch == '\t' {
        return 1;
    }
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

fn floor_char_boundary(s: &str, byte_cursor: usize) -> usize {
    let mut cursor = byte_cursor.min(s.len());
    while cursor > 0 && !s.is_char_boundary(cursor) {
        cursor -= 1;
    }
    cursor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_width_ascii() {
        assert_eq!(display_width("hi"), 2);
    }

    #[test]
    fn test_display_width_cjk() {
        assert_eq!(display_width("안녕"), 4);
    }

    #[test]
    fn test_display_width_mixed() {
        assert_eq!(display_width("안녕 hello"), 10);
    }

    #[test]
    fn test_truncate_ascii() {
        assert_eq!(truncate_display("hello world", 6), "hello…");
    }

    #[test]
    fn test_truncate_cjk() {
        assert_eq!(truncate_display("안녕하세요", 6), "안녕…");
    }

    #[test]
    fn test_truncate_no_truncation() {
        assert_eq!(truncate_display("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_max_width_lt_2() {
        assert_eq!(truncate_display("hi", 1), "");
    }

    #[test]
    fn test_wrap_mixed() {
        assert_eq!(wrap_display("한국어abc", 4), vec!["한국", "어ab", "c"]);
    }

    #[test]
    fn test_column_at_byte_zero() {
        assert_eq!(column_at_byte("안녕", 0), 0);
    }

    #[test]
    fn test_column_at_byte_after_first_cjk() {
        assert_eq!(column_at_byte("안녕", 3), 2);
    }

    #[test]
    fn test_byte_at_column() {
        assert_eq!(byte_at_column("안녕", 2), 3);
    }

    #[test]
    fn test_byte_at_column_inside_wide_char_returns_char_start() {
        assert_eq!(byte_at_column("안녕", 1), 0);
        assert_eq!(byte_at_column("a안", 2), 1);
    }

    #[test]
    fn test_column_at_byte_inside_multibyte_rounds_down_to_boundary() {
        assert_eq!(column_at_byte("안녕", 1), 0);
        assert_eq!(column_at_byte("a안", 2), 1);
    }

    #[test]
    fn test_wrap_empty() {
        assert_eq!(wrap_display("", 10), vec![""]);
    }
}
