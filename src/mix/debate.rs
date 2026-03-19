
use crate::error::Result;
use crate::r#gen::Message;
use crate::npc_compiler::Npc;

#[derive(Debug, Clone)]
pub struct DebateRound {
    pub npc_name: String,
    pub argument: String,
}

#[derive(Debug, Clone)]
pub struct DebateResult {
    pub rounds: Vec<DebateRound>,
    pub summary: String,
}

pub async fn debate(
    
    npcs: &[&Npc],
    topic: &str,
    rounds: usize,
) -> Result<DebateResult> {
    if npcs.is_empty() {
        return Ok(DebateResult {
            rounds: Vec::new(),
            summary: "No participants in debate.".to_string(),
        });
    }

    let mut debate_rounds = Vec::new();
    let mut conversation_history = String::new();

    for round_num in 0..rounds {
        for npc in npcs {
            let prompt = if conversation_history.is_empty() {
                format!(
                    "You are participating in a debate about: {topic}\n\n\
                     This is round {round} of {total_rounds}. \
                     You are {name}. Present your argument.",
                    topic = topic,
                    round = round_num + 1,
                    total_rounds = rounds,
                    name = npc.name,
                )
            } else {
                format!(
                    "You are participating in a debate about: {topic}\n\n\
                     Previous arguments:\n{history}\n\n\
                     This is round {round} of {total_rounds}. \
                     You are {name}. Respond to the previous arguments and present your position.",
                    topic = topic,
                    history = conversation_history,
                    round = round_num + 1,
                    total_rounds = rounds,
                    name = npc.name,
                )
            };

            let model = npc.resolved_model();
            let provider = npc.resolved_provider();

            let system_prompt = npc.system_prompt(None);
            let messages = vec![
                Message::system(&system_prompt),
                Message::user(&prompt),
            ];

            let response = crate::r#gen::get_genai_response(
                    &provider,
                    &model,
                    &messages,
                    None,
                    npc.api_url.as_deref(),
                )
                .await?;

            let argument = response.message.content.unwrap_or_default();

            conversation_history.push_str(&format!(
                "\n[{name} - Round {round}]: {arg}\n",
                name = npc.name,
                round = round_num + 1,
                arg = argument,
            ));

            debate_rounds.push(DebateRound {
                npc_name: npc.name.clone(),
                argument,
            });
        }
    }

    let summary = generate_summary(npcs[0], topic, &conversation_history).await?;

    Ok(DebateResult {
        rounds: debate_rounds,
        summary,
    })
}

async fn generate_summary(
    
    summarizer: &Npc,
    topic: &str,
    conversation_history: &str,
) -> Result<String> {
    let model = summarizer.resolved_model();
    let provider = summarizer.resolved_provider();

    let prompt = format!(
        "The following is a debate about: {topic}\n\n\
         {history}\n\n\
         Please provide a concise, balanced summary of the key arguments \
         presented by each participant, noting areas of agreement and disagreement.",
        topic = topic,
        history = conversation_history,
    );

    let messages = vec![
        Message::system("You are an impartial debate summarizer. Provide a balanced, concise summary."),
        Message::user(&prompt),
    ];

    let response = crate::r#gen::get_genai_response(
            &provider,
            &model,
            &messages,
            None,
            summarizer.api_url.as_deref(),
        )
        .await?;

    Ok(response.message.content.unwrap_or_default())
}
