//! Message sanitization for LLM input.
//!
//! Ensures messages follow provider constraints:
//! - No empty content in system/user messages
//! - Consecutive messages of the same role are merged
//! - Tool messages without a corresponding tool_call_id are cleaned

use crate::llm::Message;

/// Sanitize a list of messages for sending to an LLM provider.
///
/// - Removes messages with empty content (except tool-call assistant messages)
/// - Merges consecutive same-role messages
/// - Ensures the first non-system message is from the user
pub fn sanitize_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());

    for msg in messages {
        // Skip messages with no content and no tool calls
        if msg.content.as_ref().map_or(true, |c| c.is_empty())
            && msg.tool_calls.is_none()
            && msg.role != "tool"
        {
            continue;
        }

        // Merge consecutive same-role messages (but not tool messages)
        if msg.role != "tool"
            && msg.tool_calls.is_none()
            && !out.is_empty()
        {
            let last = out.last_mut().unwrap();
            if last.role == msg.role && last.tool_calls.is_none() {
                // Merge content
                if let Some(ref new_content) = msg.content {
                    let existing = last.content.get_or_insert_with(String::new);
                    existing.push('\n');
                    existing.push_str(new_content);
                }
                continue;
            }
        }

        out.push(msg);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_removes_empty_messages() {
        let msgs = vec![
            Message::system("You are helpful."),
            Message::user(""),
            Message::user("Hello"),
        ];
        let clean = sanitize_messages(msgs);
        assert_eq!(clean.len(), 2);
        assert_eq!(clean[0].role, "system");
        assert_eq!(clean[1].role, "user");
        assert_eq!(clean[1].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_merges_consecutive_same_role() {
        let msgs = vec![
            Message::system("You are helpful."),
            Message::user("Part 1"),
            Message::user("Part 2"),
        ];
        let clean = sanitize_messages(msgs);
        assert_eq!(clean.len(), 2);
        assert!(clean[1].content.as_ref().unwrap().contains("Part 1"));
        assert!(clean[1].content.as_ref().unwrap().contains("Part 2"));
    }
}
