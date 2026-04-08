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

    pub(crate) fn insert_char(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.preferred_column = None;
        self.history_index = None;
    }

    pub(crate) fn insert_newline(&mut self) {
        self.insert_char('\n');
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
            if col >= width {
                row += 1;
                col = 0;
            }
            col += 1;
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
    text[current_line_start(text, cursor)..cursor]
        .chars()
        .count()
}

fn offset_in_line(text: &str, start: usize, end: usize, target_column: usize) -> usize {
    let mut cursor = start;
    for (count, ch) in text[start..end].chars().enumerate() {
        if count >= target_column {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn wrap_text_preserving_newlines(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for raw_line in text.lines() {
        if raw_line.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut current = String::new();
        for ch in raw_line.chars() {
            if current.chars().count() >= width {
                out.push(current);
                current = String::new();
            }
            current.push(ch);
        }
        out.push(current);
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
}
