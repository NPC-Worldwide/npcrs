use npcrs::npc_compiler::load_team_from_directory;
use npcrs::r#gen::response_types::Message;

fn main() {
    let model = std::env::args().nth(1).expect("model path");
    let team_dir = std::env::args().nth(2).expect("team dir");
    let user_msg = std::env::args().nth(3).unwrap_or("what time is it".into());

    let team = load_team_from_directory(&team_dir).expect("team load");
    let mut tools: Vec<_> = team.jinxes.values().filter_map(|j| j.to_tool_def()).collect();
    tools.sort_by(|a, b| a.function.name.cmp(&b.function.name));

    let npc_name = team.forenpc.clone().unwrap_or_else(|| "sneeze".into());
    let npc = team.npcs.get(&npc_name).expect("forenpc");
    let system = npc.system_prompt(team.context.as_deref());

    let messages = vec![Message::system(system), Message::user(user_msg)];

    // Turn 1
    let resp = npcrs::r#gen::get_llamacpp_response(&model, &messages, Some(&tools), 512, 0.7, 4096, -1)
        .expect("inference");
    let mut msg = resp.message.clone();
    println!("TURN1 content: {:?}", msg.content);
    println!("TURN1 tool_calls: {:?}", msg.tool_calls.as_ref().map(|t| t.iter().map(|c| c.function.name.clone()).collect::<Vec<_>>()));

    // Turn 2: feed tool result back (simulate executor output)
    if let Some(calls) = msg.tool_calls.take() {
        let mut hist = messages.clone();
        let mut assistant = msg.clone();
        assistant.tool_calls = Some(calls.clone());
        hist.push(assistant);
        for c in &calls {
            let mut tr = Message::tool_result(&c.id, "\"15:47 PM, Tuesday, August 25\"");
            tr.name = Some(c.function.name.clone());
            hist.push(tr);
        }
        let resp2 = npcrs::r#gen::get_llamacpp_response(&model, &hist, Some(&tools), 512, 0.7, 4096, -1)
            .expect("inference2");
        let m2 = resp2.message.clone();
        println!("TURN2 content: {:?}", m2.content);
        println!("TURN2 tool_calls: {:?}", m2.tool_calls.as_ref().map(|t| t.iter().map(|c| c.function.name.clone()).collect::<Vec<_>>()));
    }
}
