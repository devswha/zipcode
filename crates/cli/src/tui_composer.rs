#[derive(Debug, Default, Clone)]
pub(crate) struct Composer {
    buffer: String,
    cursor: usize,
    preferred_column: Option<usize>,
    history: Vec<String>,
    history_index: Option<usize>,
}

impl Composer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub(crate) fn text(&self) -> &str {
        &self.buffer
    }

    pub(crate) fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
        self.preferred_column = None;
        self.history_index = None;
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub(crate) fn insert_char(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.preferred_column = None;
        self.history_index = None;
    }

    pub(crate) fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    pub(crate) fn try_escape_newline(&mut self) -> bool {
        if self.cursor != self.buffer.len() || self.cursor == 0 {
            return false;
        }

        let prev = prev_boundary(&self.buffer, self.cursor);
        if &self.buffer[prev..self.cursor] != "\\" {
            return false;
        }

        self.buffer.drain(prev..self.cursor);
        self.cursor = prev;
        self.insert_newline();
        true
    }

    pub(crate) fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = prev_boundary(&self.buffer, self.cursor);
        self.buffer.drain(prev..self.cursor);
        self.cursor = prev;
        self.preferred_column = None;
        self.history_index = None;
    }

    pub(crate) fn delete_forward(&mut self) {
        if self.cursor >= self.buffer.len() {
            return;
        }
        let next = next_boundary(&self.buffer, self.cursor);
        self.buffer.drain(self.cursor..next);
        self.preferred_column = None;
        self.history_index = None;
    }

    pub(crate) fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = prev_boundary(&self.buffer, self.cursor);
            self.preferred_column = None;
        }
    }

    pub(crate) fn move_right(&mut self) {
        if self.cursor < self.buffer.len() {
            self.cursor = next_boundary(&self.buffer, self.cursor);
            self.preferred_column = None;
        }
    }

    pub(crate) fn move_home(&mut self) {
        self.cursor = current_line_start(&self.buffer, self.cursor);
        self.preferred_column = None;
    }

    pub(crate) fn move_end(&mut self) {
        self.cursor = current_line_end(&self.buffer, self.cursor);
        self.preferred_column = None;
    }

    pub(crate) fn move_up(&mut self) -> bool {
        let line_start = current_line_start(&self.buffer, self.cursor);
        if line_start == 0 {
            return false;
        }

        let column = self
            .preferred_column
            .unwrap_or_else(|| current_column(&self.buffer, self.cursor));
        let prev_end = line_start.saturating_sub(1);
        let prev_start = current_line_start(&self.buffer, prev_end);
        self.cursor = offset_in_line(&self.buffer, prev_start, prev_end, column);
        self.preferred_column = Some(column);
        true
    }

    pub(crate) fn move_down(&mut self) -> bool {
        let line_end = current_line_end(&self.buffer, self.cursor);
        if line_end >= self.buffer.len() {
            return false;
        }

        let column = self
            .preferred_column
            .unwrap_or_else(|| current_column(&self.buffer, self.cursor));
        let next_start = next_boundary(&self.buffer, line_end);
        let next_end = current_line_end(&self.buffer, next_start);
        self.cursor = offset_in_line(&self.buffer, next_start, next_end, column);
        self.preferred_column = Some(column);
        true
    }

    #[cfg(test)]
    pub(crate) fn can_move_up(&self) -> bool {
        current_line_start(&self.buffer, self.cursor) > 0
    }

    #[cfg(test)]
    pub(crate) fn can_move_down(&self) -> bool {
        current_line_end(&self.buffer, self.cursor) < self.buffer.len()
    }

    pub(crate) fn submit(&mut self) -> String {
        let submitted = self.buffer.clone();
        if !submitted.trim().is_empty() {
            self.history.push(submitted.clone());
        }
        self.clear();
        submitted
    }

    pub(crate) fn history_previous(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        let next_index = match self.history_index {
            Some(index) if index > 0 => index - 1,
            Some(_) => 0,
            None => self.history.len() - 1,
        };
        self.load_history(next_index);
        true
    }

    pub(crate) fn history_next(&mut self) -> bool {
        let Some(index) = self.history_index else {
            return false;
        };
        if index + 1 >= self.history.len() {
            self.clear();
            return true;
        }
        self.load_history(index + 1);
        true
    }

    pub(crate) fn wrapped_lines(&self, width: usize) -> Vec<String> {
        wrap_text_preserving_newlines(&self.buffer, width)
    }

    pub(crate) fn cursor_visual_position(&self, width: usize) -> (usize, usize) {
        let width = width.max(1);
        let mut row = 0usize;
        let mut col = 0usize;
        for ch in self.buffer[..self.cursor].chars() {
            if ch == '\n' {
                row += 1;
                col = 0;
                continue;
            }
            let w = crate::width::char_display_width(ch);
            if col > 0 && col + w > width {
                row += 1;
                col = 0;
            }
            col += w;
        }
        if col >= width {
            row += 1;
            col = 0;
        }
        (row, col)
    }

    fn load_history(&mut self, index: usize) {
        self.history_index = Some(index);
        self.buffer = self.history[index].clone();
        self.cursor = self.buffer.len();
        self.preferred_column = None;
    }
}

fn prev_boundary(text: &str, cursor: usize) -> usize {
    text[..cursor]
        .char_indices()
        .last()
        .map_or(0, |(idx, _)| idx)
}

fn next_boundary(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .char_indices()
        .nth(1)
        .map_or(text.len(), |(offset, _)| cursor + offset)
}

fn current_line_start(text: &str, cursor: usize) -> usize {
    text[..cursor].rfind('\n').map_or(0, |idx| idx + 1)
}

fn current_line_end(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .find('\n')
        .map_or(text.len(), |offset| cursor + offset)
}

fn current_column(text: &str, cursor: usize) -> usize {
    let line_start = current_line_start(text, cursor);
    crate::width::column_at_byte(&text[line_start..], cursor - line_start)
}

fn offset_in_line(text: &str, start: usize, end: usize, target_column: usize) -> usize {
    start + crate::width::byte_at_column(&text[start..end], target_column)
}

fn wrap_text_preserving_newlines(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for raw_line in text.lines() {
        if raw_line.is_empty() {
            out.push(String::new());
            continue;
        }
        out.extend(crate::width::wrap_display(raw_line, width));
    }
    if text.ends_with('\n') {
        out.push(String::new());
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiline_navigation_moves_between_lines() {
        let mut composer = Composer::new();
        for ch in "abc\ndef".chars() {
            composer.insert_char(ch);
        }
        composer.move_home();
        composer.move_down();
        assert_eq!(composer.text(), "abc\ndef");
        assert!(composer.can_move_up());
        assert!(!composer.can_move_down());
    }

    #[test]
    fn history_round_trip_restores_submitted_entries() {
        let mut composer = Composer::new();
        for ch in "first".chars() {
            composer.insert_char(ch);
        }
        assert_eq!(composer.submit(), "first");
        for ch in "second".chars() {
            composer.insert_char(ch);
        }
        assert_eq!(composer.submit(), "second");

        assert!(composer.history_previous());
        assert_eq!(composer.text(), "second");
        assert!(composer.history_previous());
        assert_eq!(composer.text(), "first");
        assert!(composer.history_next());
        assert_eq!(composer.text(), "second");
    }

    #[test]
    fn cursor_visual_position_tracks_wrapped_multiline_text() {
        let mut composer = Composer::new();
        for ch in "abcd\nef".chars() {
            composer.insert_char(ch);
        }
        let (row, col) = composer.cursor_visual_position(3);
        assert_eq!((row, col), (2, 2));
    }

    #[test]
    fn cursor_visual_position_cjk_counts_double_width() {
        let mut composer = Composer::new();
        // "안녕" → 4 display columns (2+2)
        for ch in "안녕".chars() {
            composer.insert_char(ch);
        }
        let (row, col) = composer.cursor_visual_position(10);
        assert_eq!((row, col), (0, 4));
    }

    #[test]
    fn wrapped_lines_cjk_wraps_by_display_width() {
        let composer = {
            let mut c = Composer::new();
            // "한국어" → 6 display columns (2+2+2)
            for ch in "한국어".chars() {
                c.insert_char(ch);
            }
            c
        };
        // width=3 → each CJK char is 2 cols, so only 1 char fits per line
        // (second char would need 2+2=4 > 3)
        let lines = composer.wrapped_lines(3);
        assert_eq!(lines, vec!["한", "국", "어"]);
    }

    #[test]
    fn wrapped_lines_cjk_mixed_width() {
        let composer = {
            let mut c = Composer::new();
            for ch in "한국어abc".chars() {
                c.insert_char(ch);
            }
            c
        };
        // width=4: "한국"=4, "어ab"=2+1+1=4, "c"=1
        let lines = composer.wrapped_lines(4);
        assert_eq!(lines, vec!["한국", "어ab", "c"]);
    }

    #[test]
    fn move_up_down_cjk_preserves_visual_column() {
        let mut composer = Composer::new();
        // Line 1: "abcd" (4 cols), Line 2: "안녕하" (6 cols)
        for ch in "abcd\n안녕하".chars() {
            composer.insert_char(ch);
        }
        // Cursor is at end of line 2 (col 6 in display width)
        assert!(composer.can_move_up());

        // Move to start of current line
        composer.move_home();
        // Now at start of line 2 (col 0)
        assert!(composer.can_move_up());
        assert!(!composer.can_move_down());

        // Move up — should go to line 1, col 0
        assert!(composer.move_up());
        assert!(!composer.can_move_up());
        assert!(composer.can_move_down());
    }

    #[test]
    fn move_down_clamps_to_start_of_wide_char_when_target_column_is_inside_it() {
        let mut composer = Composer::new();
        for ch in "a\n안녕".chars() {
            composer.insert_char(ch);
        }

        assert!(composer.move_up());
        assert_eq!(composer.cursor, 1); // end of first line (display col 1)
        composer.move_end();
        assert!(composer.move_down());
        assert_eq!(composer.cursor, "a\n".len()); // clamp to start of '안'
    }

    #[test]
    fn cursor_visual_position_wide_char_in_single_column_area_avoids_blank_row() {
        let mut composer = Composer::new();
        composer.insert_char('한');

        assert_eq!(composer.cursor_visual_position(1), (1, 0));
    }

    #[test]
    fn trailing_backslash_then_enter_inserts_newline() {
        let mut composer = Composer::new();
        for ch in "hello\\".chars() {
            composer.insert_char(ch);
        }

        assert!(composer.try_escape_newline());
        assert_eq!(composer.text(), "hello\n");
    }

    // ── New tests: backspace edge cases ────────────────────────────────

    #[test]
    fn backspace_at_beginning_is_noop() {
        let mut composer = Composer::new();
        assert_eq!(composer.cursor, 0);
        composer.backspace();
        assert_eq!(composer.text(), "");
        assert_eq!(composer.cursor, 0, "cursor must stay at 0 on empty buffer");
    }

    #[test]
    fn backspace_deletes_last_ascii_char() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.backspace();
        assert_eq!(composer.text(), "ab");
        assert_eq!(composer.cursor, 2);
    }

    #[test]
    fn backspace_deletes_from_middle() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.move_left(); // cursor at 'c' boundary (index 2)
        composer.backspace();
        assert_eq!(composer.text(), "ac");
        assert_eq!(
            composer.cursor, 1,
            "cursor should be at index 1 (after 'a')"
        );
    }

    #[test]
    fn backspace_deletes_multibyte_utf8_char() {
        let mut composer = Composer::new();
        for ch in "ab녕".chars() {
            composer.insert_char(ch);
        }
        // '녕' is 3 bytes in UTF-8
        composer.backspace();
        assert_eq!(composer.text(), "ab");
        assert_eq!(composer.cursor, 2);
    }

    #[test]
    fn backspace_across_newline() {
        let mut composer = Composer::new();
        for ch in "ab\ncd".chars() {
            composer.insert_char(ch);
        }
        composer.move_left();
        composer.move_left();
        // cursor is between \n and 'c'
        composer.backspace();
        assert_eq!(composer.text(), "abcd");
    }

    // ── New tests: delete_forward edge cases ───────────────────────────

    #[test]
    fn delete_forward_at_end_is_noop() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        assert_eq!(composer.cursor, 3);
        composer.delete_forward();
        assert_eq!(composer.text(), "abc");
        assert_eq!(composer.cursor, 3);
    }

    #[test]
    fn delete_forward_removes_char_after_cursor() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.move_left(); // cursor at index 2, before 'c'
        composer.delete_forward();
        assert_eq!(composer.text(), "ab");
        assert_eq!(composer.cursor, 2);
    }

    #[test]
    fn delete_forward_removes_multibyte_char() {
        let mut composer = Composer::new();
        for ch in "a한b".chars() {
            composer.insert_char(ch);
        }
        // cursor at end (index 5: 1 + 3 + 1)
        composer.move_left(); // before 'b'
        composer.move_left(); // before '한'
        composer.delete_forward();
        assert_eq!(composer.text(), "ab");
        assert_eq!(composer.cursor, 1);
    }

    #[test]
    fn delete_forward_on_empty_buffer_is_noop() {
        let mut composer = Composer::new();
        composer.delete_forward();
        assert_eq!(composer.text(), "");
        assert_eq!(composer.cursor, 0);
    }

    // ── New tests: clear ───────────────────────────────────────────────

    #[test]
    fn clear_resets_buffer_cursor_and_history_index() {
        let mut composer = Composer::new();
        for ch in "hello".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();
        for ch in "world".chars() {
            composer.insert_char(ch);
        }
        assert!(composer.history_previous());

        composer.clear();
        assert_eq!(composer.text(), "");
        assert_eq!(composer.cursor, 0);
        assert!(composer.is_empty());
        // history should still be intact (clear doesn't wipe history)
        assert!(composer.history_previous(), "history should survive clear");
    }

    // ── New tests: is_empty ────────────────────────────────────────────

    #[test]
    fn is_empty_on_new_composer() {
        let composer = Composer::new();
        assert!(composer.is_empty());
    }

    #[test]
    fn is_empty_false_after_insert() {
        let mut composer = Composer::new();
        composer.insert_char('x');
        assert!(!composer.is_empty());
    }

    #[test]
    fn is_empty_true_after_clear() {
        let mut composer = Composer::new();
        composer.insert_char('x');
        composer.clear();
        assert!(composer.is_empty());
    }

    #[test]
    fn is_empty_false_after_insert_newline() {
        let mut composer = Composer::new();
        composer.insert_newline();
        assert!(!composer.is_empty(), "newline is content, not empty");
    }

    // ── New tests: insert_newline ──────────────────────────────────────

    #[test]
    fn insert_newline_creates_multiline_buffer() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.insert_newline();
        for ch in "def".chars() {
            composer.insert_char(ch);
        }
        assert_eq!(composer.text(), "abc\ndef");
        assert_eq!(composer.cursor, 7);
    }

    #[test]
    fn insert_newline_at_beginning() {
        let mut composer = Composer::new();
        composer.insert_newline();
        assert_eq!(composer.text(), "\n");
        assert_eq!(composer.cursor, 1);
    }

    // ── New tests: move_left / move_right ──────────────────────────────

    #[test]
    fn move_left_on_empty_is_noop() {
        let mut composer = Composer::new();
        composer.move_left();
        assert_eq!(composer.cursor, 0);
    }

    #[test]
    fn move_left_decrements_by_char() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.move_left();
        assert_eq!(composer.cursor, 2);
        composer.move_left();
        assert_eq!(composer.cursor, 1);
        composer.move_left();
        assert_eq!(composer.cursor, 0);
        composer.move_left(); // noop at start
        assert_eq!(composer.cursor, 0);
    }

    #[test]
    fn move_left_across_multibyte() {
        let mut composer = Composer::new();
        for ch in "a한b".chars() {
            composer.insert_char(ch);
        }
        // cursor at end = 5 (1 + 3 + 1)
        composer.move_left(); // before 'b' = 4
        assert_eq!(composer.cursor, 4);
        composer.move_left(); // before '한' = 1
        assert_eq!(composer.cursor, 1);
    }

    #[test]
    fn move_right_on_empty_is_noop() {
        let mut composer = Composer::new();
        composer.move_right();
        assert_eq!(composer.cursor, 0);
    }

    #[test]
    fn move_right_at_end_is_noop() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.move_right();
        assert_eq!(composer.cursor, 3, "cursor must stay at end");
    }

    #[test]
    fn move_right_advances_by_char() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.move_home(); // cursor at 0
        composer.move_right();
        assert_eq!(composer.cursor, 1);
        composer.move_right();
        assert_eq!(composer.cursor, 2);
    }

    // ── New tests: move_home / move_end ────────────────────────────────

    #[test]
    fn move_home_on_single_line() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.move_home();
        assert_eq!(composer.cursor, 0);
    }

    #[test]
    fn move_end_on_single_line() {
        let mut composer = Composer::new();
        for ch in "abc".chars() {
            composer.insert_char(ch);
        }
        composer.move_home();
        composer.move_end();
        assert_eq!(composer.cursor, 3);
    }

    #[test]
    fn move_home_goes_to_current_line_start_multiline() {
        let mut composer = Composer::new();
        for ch in "abc\ndef".chars() {
            composer.insert_char(ch);
        }
        // cursor at end of "def" = 7
        composer.move_home();
        assert_eq!(
            composer.cursor, 4,
            "cursor should be at start of second line (after \\n)"
        );
    }

    #[test]
    fn move_end_goes_to_current_line_end_multiline() {
        let mut composer = Composer::new();
        for ch in "abc\ndef".chars() {
            composer.insert_char(ch);
        }
        // cursor at end = 7
        composer.move_home(); // cursor at 4 (start of "def")
        composer.move_end();
        assert_eq!(composer.cursor, 7, "cursor should be at end of second line");
    }

    #[test]
    fn move_home_on_first_line_of_multiline() {
        let mut composer = Composer::new();
        for ch in "abc\ndef".chars() {
            composer.insert_char(ch);
        }
        // Move to first line
        composer.move_home(); // at 4
        assert!(composer.move_up()); // now on first line
        composer.move_end(); // at 3 (end of "abc")
        composer.move_home();
        assert_eq!(composer.cursor, 0, "home on first line should be index 0");
    }

    // ── New tests: submit edge cases ───────────────────────────────────

    #[test]
    fn submit_clears_buffer_after_returning() {
        let mut composer = Composer::new();
        for ch in "test".chars() {
            composer.insert_char(ch);
        }
        let result = composer.submit();
        assert_eq!(result, "test");
        assert_eq!(composer.text(), "");
        assert_eq!(composer.cursor, 0);
        assert!(composer.is_empty());
    }

    #[test]
    fn submit_stores_non_empty_in_history() {
        let mut composer = Composer::new();
        for ch in "hello".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();
        assert!(composer.history_previous());
        assert_eq!(composer.text(), "hello");
    }

    #[test]
    fn submit_does_not_store_whitespace_only_in_history() {
        let mut composer = Composer::new();
        for ch in "   ".chars() {
            composer.insert_char(ch);
        }
        let result = composer.submit();
        assert_eq!(result, "   ");
        assert!(
            !composer.history_previous(),
            "whitespace-only should not be stored"
        );
    }

    #[test]
    fn submit_does_not_store_empty_in_history() {
        let mut composer = Composer::new();
        let result = composer.submit();
        assert_eq!(result, "");
        assert!(
            !composer.history_previous(),
            "empty submit should not be stored"
        );
    }

    // ── New tests: history_previous / history_next edge cases ──────────

    #[test]
    fn history_previous_on_empty_history_returns_false() {
        let mut composer = Composer::new();
        assert!(!composer.history_previous());
    }

    #[test]
    fn history_next_without_previous_returns_false() {
        let mut composer = Composer::new();
        for ch in "test".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();
        assert!(
            !composer.history_next(),
            "history_next without history_previous should be false"
        );
    }

    #[test]
    fn history_next_at_end_clears_buffer() {
        let mut composer = Composer::new();
        for ch in "first".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();
        for ch in "second".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();

        // Go back to oldest
        assert!(composer.history_previous()); // "second"
        assert!(composer.history_previous()); // "first" (oldest)
                                              // history_next past end should clear
        assert!(composer.history_next()); // "second"
        assert!(composer.history_next()); // past end → clear
        assert_eq!(composer.text(), "");
        assert!(composer.is_empty());
    }

    #[test]
    fn history_previous_clamps_at_oldest_entry() {
        let mut composer = Composer::new();
        for ch in "first".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();
        for ch in "second".chars() {
            composer.insert_char(ch);
        }
        let _ = composer.submit();

        assert!(composer.history_previous()); // "second"
        assert!(composer.history_previous()); // "first"
        assert!(composer.history_previous()); // still "first" (clamped)
        assert_eq!(composer.text(), "first");
    }

    // ── New tests: try_escape_newline edge cases ───────────────────────

    #[test]
    fn try_escape_newline_not_at_end_returns_false() {
        let mut composer = Composer::new();
        for ch in "a\\b".chars() {
            composer.insert_char(ch);
        }
        composer.move_home();
        // cursor at start, not at end
        assert!(!composer.try_escape_newline());
        assert_eq!(composer.text(), "a\\b");
    }

    #[test]
    fn try_escape_newline_no_trailing_backslash_returns_false() {
        let mut composer = Composer::new();
        for ch in "hello".chars() {
            composer.insert_char(ch);
        }
        assert!(!composer.try_escape_newline());
        assert_eq!(composer.text(), "hello");
    }

    #[test]
    fn try_escape_newline_on_empty_buffer_returns_false() {
        let mut composer = Composer::new();
        assert!(!composer.try_escape_newline());
    }

    #[test]
    fn try_escape_newline_double_backslash_becomes_single_newline() {
        let mut composer = Composer::new();
        // "\\\\" in Rust source = two backslashes in the string
        // The last one should be treated as the escape backslash
        for ch in "\\\\".chars() {
            composer.insert_char(ch);
        }
        assert!(composer.try_escape_newline());
        // First backslash remains, trailing backslash replaced with \n
        assert_eq!(composer.text(), "\\\n");
    }

    // ── New tests: wrapped_lines edge cases ────────────────────────────

    #[test]
    fn wrapped_lines_empty_buffer() {
        let composer = Composer::new();
        let lines = composer.wrapped_lines(10);
        assert_eq!(lines, vec![""]);
    }

    #[test]
    fn wrapped_lines_trailing_newline_produces_extra_empty_line() {
        let mut composer = Composer::new();
        for ch in "abc\n".chars() {
            composer.insert_char(ch);
        }
        let lines = composer.wrapped_lines(10);
        assert_eq!(lines, vec!["abc", ""]);
    }

    #[test]
    fn wrapped_lines_consecutive_empty_lines() {
        let mut composer = Composer::new();
        for ch in "a\n\nb".chars() {
            composer.insert_char(ch);
        }
        let lines = composer.wrapped_lines(10);
        assert_eq!(lines, vec!["a", "", "b"]);
    }

    // ── New tests: cursor_visual_position edge cases ───────────────────

    #[test]
    fn cursor_visual_position_on_empty_buffer() {
        let composer = Composer::new();
        assert_eq!(composer.cursor_visual_position(10), (0, 0));
    }

    #[test]
    fn cursor_visual_position_single_char() {
        let mut composer = Composer::new();
        composer.insert_char('x');
        assert_eq!(composer.cursor_visual_position(10), (0, 1));
    }

    #[test]
    fn cursor_visual_position_wraps_at_width() {
        let mut composer = Composer::new();
        for ch in "abcdef".chars() {
            composer.insert_char(ch);
        }
        // 6 chars at width 4 → row 0: "abcd", row 1: "ef" → cursor at (1, 2)
        assert_eq!(composer.cursor_visual_position(4), (1, 2));
    }

    #[test]
    fn cursor_visual_position_single_char_at_zero_width() {
        let mut composer = Composer::new();
        composer.insert_char('a');
        // width is clamped to max(1, 0) = 1. 'a' fills exactly 1 column,
        // so col >= width triggers a wrap: result is (1, 0).
        assert_eq!(composer.cursor_visual_position(0), (1, 0));
    }
}
