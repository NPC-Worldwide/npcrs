
use crate::r#gen::Message;

pub fn sanitize_messages(messages: Vec<Message>) -> Vec<Message> {
    if messages.is_empty() {
        return messages;
    }

    let mut cleaned: Vec<Message> = Vec::with_capacity(messages.len());
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];

        if msg.role == "assistant" && msg.tool_calls.is_some() {
            let tool_calls = msg.tool_calls.as_ref().unwrap();
            let expected_ids: std::collections::HashSet<&str> = tool_calls
                .iter()
                .map(|tc| tc.id.as_str())
                .collect();

            let mut fulfilled_ids = std::collections::HashSet::new();
            let mut j = i + 1;
            while j < messages.len() && messages[j].role == "tool" {
                if let Some(ref tid) = messages[j].tool_call_id {
                    fulfilled_ids.insert(tid.as_str());
                }
                j += 1;
            }

            if !expected_ids.is_empty() && expected_ids.is_subset(&fulfilled_ids) {
                cleaned.push(messages[i].clone());
                for k in (i + 1)..j {
                    cleaned.push(messages[k].clone());
                }
            } else {
                if let Some(ref text) = msg.content {
                    if !text.is_empty() {
                        cleaned.push(Message::assistant(text));
                    }
                }
            }
            i = j;
        } else if msg.role == "tool" {
            let content = msg.content.as_deref().unwrap_or("");
            let name = msg.name.as_deref().unwrap_or("tool");
            if name != "tool" {
                cleaned.push(Message::assistant(format!("[{} result]: {}", name, content)));
            } else {
                cleaned.push(Message::assistant(content));
            }
            i += 1;
        } else {
            cleaned.push(messages[i].clone());
            i += 1;
        }
    }

    let mut merged: Vec<Message> = Vec::with_capacity(cleaned.len());
    for msg in cleaned {
        let dominated = !merged.is_empty()
            && (msg.role == "user" || msg.role == "assistant")
            && msg.tool_calls.is_none()
            && merged.last().map_or(false, |last: &Message| {
                last.role == msg.role && last.tool_calls.is_none()
            });

        if dominated {
            let last = merged.last_mut().unwrap();
            let prev = last.content.take().unwrap_or_default();
            let new = msg.content.unwrap_or_default();
            last.content = Some(format!("{}\n{}", prev, new).trim().to_string());
        } else {
            merged.push(msg);
        }
    }

    while merged
        .last()
        .map_or(false, |m| m.role == "assistant" && m.tool_calls.is_none())
    {
        merged.pop();
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#gen::{ToolCall, ToolCallFunction};

    #[test]
    fn test_removes_empty_messages() {
        let msgs = vec![
            Message::system("You are helpful."),
            Message::user(""),
            Message::user("Hello"),
        ];
        let clean = sanitize_messages(msgs);
        assert_eq!(clean.len(), 2);
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

    #[test]
    fn test_strips_trailing_assistant() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant("hello"),
        ];
        let clean = sanitize_messages(msgs);
        assert_eq!(clean.len(), 2);
        assert_eq!(clean[1].role, "user");
    }

    #[test]
    fn test_orphaned_tool_calls_stripped() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("hi"),
            Message {
                role: "assistant".into(),
                content: Some("Let me check.".into()),
                tool_calls: Some(vec![ToolCall {
                    id: "tc_1".into(),
                    r#type: "function".into(),
                    function: ToolCallFunction {
                        name: "sh".into(),
                        arguments: "{}".into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
        ];
        let clean = sanitize_messages(msgs);
        assert_eq!(clean.len(), 2);
    }

    #[test]
    fn test_valid_tool_pair_kept() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("hi"),
            Message {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "tc_1".into(),
                    r#type: "function".into(),
                    function: ToolCallFunction {
                        name: "sh".into(),
                        arguments: "{}".into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            Message::tool_result("tc_1", "output"),
        ];
        let clean = sanitize_messages(msgs);
        assert_eq!(clean.len(), 4);
        assert!(clean[2].tool_calls.is_some());
        assert_eq!(clean[3].role, "tool");
    }
}
