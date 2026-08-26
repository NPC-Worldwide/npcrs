fn main() {
    let dir = std::env::args().nth(1).expect("usage: teamload <dir>");
    match npcrs::npc_compiler::load_team_from_directory(&dir) {
        Ok(team) => {
            println!("OK npcs={:?}", team.npcs.keys().collect::<Vec<_>>());
            println!("OK jinxes={:?}", team.jinxes.keys().collect::<Vec<_>>());
        }
        Err(e) => println!("ERR {e:?}"),
    }
}
