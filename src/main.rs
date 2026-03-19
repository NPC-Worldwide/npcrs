
use npcrs::error::Result;
use npcrs::kernel::Kernel;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{CompletionType, Config, Editor, Helper};
use std::borrow::Cow;

const CYAN: &str = "\x1b[36m";
const PURPLE: &str = "\x1b[35m";
const DIM: &str = "\x1b[90m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

struct NpcHelper {
    npc_names: Vec<String>,
    commands: Vec<String>,
}

impl NpcHelper {
    fn new(npc_names: Vec<String>, jinx_names: Vec<String>) -> Self {
        let mut commands = vec![
            "/ps", "/stats", "/help", "/quit", "/exit", "/clear",
            "/agent", "/chat", "/cmd", "/switch", "/kill", "/jinxes",
            "/set", "/history",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();

        for j in jinx_names {
            commands.push(format!("/{}", j));
        }

        Self { npc_names, commands }
    }
}

impl Completer for NpcHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let word_start = line[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
        let word = &line[word_start..pos];

        let mut matches = Vec::new();

        if word.starts_with('@') {
            let prefix = &word[1..];
            for name in &self.npc_names {
                if name.starts_with(prefix) {
                    matches.push(Pair {
                        display: format!("@{}", name),
                        replacement: format!("@{} ", name),
                    });
                }
            }
        } else if word.starts_with('/') {
            for cmd in &self.commands {
                if cmd.starts_with(word) {
                    matches.push(Pair {
                        display: cmd.clone(),
                        replacement: format!("{} ", cmd),
                    });
                }
            }
        }

        Ok((word_start, matches))
    }
}

impl Hinter for NpcHelper {
    type Hint = String;
}

impl Highlighter for NpcHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Borrowed(prompt)
    }
}

impl Validator for NpcHelper {}
impl Helper for NpcHelper {}

#[derive(Clone, PartialEq)]
enum Mode {
    Agent,
    Chat,
    Cmd,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Agent => write!(f, "agent"),
            Mode::Chat => write!(f, "chat"),
            Mode::Cmd => write!(f, "cmd"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("npcrs=warn".parse().unwrap()),
        )
        .with_target(false)
        .without_time()
        .init();

    let _ = dotenvy::dotenv();
    load_npcshrc();

    let team_dir = find_team_dir();
    let db_path = shellexpand::tilde("~/npcsh_history.db").to_string();

    let mut kernel = Kernel::boot(&team_dir, &db_path)?;

    print_welcome(&kernel);

    let config = Config::builder()
        .completion_type(CompletionType::List)
        .build();

    let npc_names: Vec<String> = kernel.ps().iter().map(|p| p.npc.name.clone()).collect();
    let jinx_names: Vec<String> = kernel.jinx_names().into_iter().map(String::from).collect();
    let helper = NpcHelper::new(npc_names, jinx_names);

    let history_path = shellexpand::tilde("~/.npcsh_history").to_string();
    let mut rl = Editor::with_config(config).unwrap();
    rl.set_helper(Some(helper));
    let _ = rl.load_history(&history_path);

    let mut current_pid: u32 = 0;
    let mut mode = Mode::Agent;
    let mut turn_count: u64 = 0;

    loop {
        let npc_name = kernel
            .get_process(current_pid)
            .map(|p| p.npc.name.as_str())
            .unwrap_or("???");

        let cwd = std::env::current_dir()
            .map(|p| {
                let s = p.display().to_string();
                let home = shellexpand::tilde("~").to_string();
                if let Some(rest) = s.strip_prefix(&home) {
                    format!("~{}", rest)
                } else {
                    s
                }
            })
            .unwrap_or_else(|_| "?".to_string());

        let model = kernel
            .get_process(current_pid)
            .map(|p| p.npc.resolved_model())
            .unwrap_or_else(|| "?".to_string());

        let prompt = format!(
            "{DIM}{cwd}{RESET} {CYAN}{BOLD}{npc_name}{RESET} {DIM}[{mode}|{model}]{RESET}\n{PURPLE}>{RESET} "
        );

        let input = match rl.readline(&prompt) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => {
                eprintln!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        };

        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }

        rl.add_history_entry(&input).ok();

        let handled = match input.as_str() {
            "exit" | "quit" | "/quit" | "/exit" => break,

            "/ps" => {
                for p in kernel.ps() {
                    let state_color = match p.state {
                        npcrs::process::ProcessState::Running => GREEN,
                        npcrs::process::ProcessState::Blocked => YELLOW,
                        npcrs::process::ProcessState::Dead => RED,
                        _ => DIM,
                    };
                    println!(
                        "  {CYAN}@{:<12}{RESET} pid:{:<3} {state_color}{:?}{RESET}  tokens:{}/{} cost:${:.4} turns:{}",
                        p.npc.name, p.pid, p.state,
                        p.usage.total_input_tokens, p.usage.total_output_tokens,
                        p.usage.total_cost_usd, p.usage.total_turns,
                    );
                }
                true
            }

            "/stats" => {
                let s = kernel.stats();
                println!(
                    "{BOLD}Kernel Stats{RESET}\n  uptime: {}s\n  processes: {} (run:{} blk:{} dead:{})\n  tokens: {} (in+out)\n  cost: ${:.4}\n  jinxes: {}",
                    s.uptime_secs, s.total_processes, s.running, s.blocked, s.dead,
                    s.total_tokens, s.total_cost_usd, s.jinx_count,
                );
                true
            }

            "/help" => {
                println!("{BOLD}npcsh-rs{RESET} — NPC OS Shell v{}\n", env!("CARGO_PKG_VERSION"));
                println!("{BOLD}Modes:{RESET}");
                println!("  {CYAN}/agent{RESET}          Full agent mode (tools + bash + LLM)");
                println!("  {CYAN}/chat{RESET}           Chat-only mode (LLM, no tools)");
                println!("  {CYAN}/cmd{RESET}            Command mode (bash first, LLM fallback)");
                println!();
                println!("{BOLD}NPC Commands:{RESET}");
                println!("  {CYAN}@npc{RESET}            Switch to NPC process");
                println!("  {CYAN}@npc command{RESET}    Delegate command to NPC");
                println!("  {CYAN}/switch <npc>{RESET}   Switch to NPC process");
                println!("  {CYAN}/kill{RESET}           Kill current process");
                println!();
                println!("{BOLD}Info:{RESET}");
                println!("  {CYAN}/ps{RESET}             List processes");
                println!("  {CYAN}/stats{RESET}          Kernel stats");
                println!("  {CYAN}/jinxes{RESET}         List available tools");
                println!("  {CYAN}/history{RESET}        Show conversation history");
                println!();
                println!("{BOLD}Config:{RESET}");
                println!("  {CYAN}/set key=val{RESET}    Set model, provider, mode");
                println!("  {CYAN}/clear{RESET}          Clear conversation");
                println!();
                println!("{BOLD}Shell:{RESET}");
                println!("  Any text is sent to the current NPC.");
                println!("  In {CYAN}/cmd{RESET} mode, input runs as bash first.");
                println!("  Tab completes @npcs and /commands.");
                true
            }

            "/agent" => {
                mode = Mode::Agent;
                eprintln!("{GREEN}Switched to agent mode{RESET}");
                true
            }
            "/chat" => {
                mode = Mode::Chat;
                eprintln!("{GREEN}Switched to chat mode{RESET}");
                true
            }
            "/cmd" => {
                mode = Mode::Cmd;
                eprintln!("{GREEN}Switched to cmd mode{RESET}");
                true
            }

            "/jinxes" => {
                let names = kernel.jinx_names();
                let mut sorted: Vec<&str> = names;
                sorted.sort();
                println!("{BOLD}Available jinxes ({}):{RESET}", sorted.len());
                for chunk in sorted.chunks(6) {
                    println!("  {}", chunk.iter().map(|n| format!("{CYAN}/{n}{RESET}")).collect::<Vec<_>>().join("  "));
                }
                true
            }

            "/clear" => {
                if let Some(p) = kernel.get_process_mut(current_pid) {
                    p.messages.clear();
                    eprintln!("{GREEN}Conversation cleared{RESET}");
                }
                true
            }

            "/history" => {
                if let Some(p) = kernel.get_process(current_pid) {
                    if p.messages.is_empty() {
                        println!("{DIM}(no messages){RESET}");
                    } else {
                        for m in &p.messages {
                            let role_color = match m.role.as_str() {
                                "user" => CYAN,
                                "assistant" => GREEN,
                                _ => DIM,
                            };
                            let content = m.content.as_deref().unwrap_or("");
                            let preview = if content.len() > 80 {
                                format!("{}...", &content[..80])
                            } else {
                                content.to_string()
                            };
                            println!("  {role_color}{:<10}{RESET} {}", m.role, preview);
                        }
                    }
                }
                true
            }

            "/kill" => {
                if current_pid == 0 {
                    eprintln!("{RED}Cannot kill init (pid 0){RESET}");
                } else {
                    let name = kernel.get_process(current_pid).map(|p| p.npc.name.clone());
                    kernel.kill(current_pid, 0).ok();
                    current_pid = 0;
                    eprintln!("{YELLOW}Killed @{} — switched to init{RESET}", name.unwrap_or_default());
                }
                true
            }

            _ => false,
        };

        if handled {
            continue;
        }

        if input.starts_with("/set ") {
            let rest = input.strip_prefix("/set ").unwrap().trim();
            handle_set_command(rest, &mut kernel, current_pid, &mut mode);
            continue;
        }

        if input.starts_with('@') {
            let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
            let target = parts[0];

            if let Some(command) = parts.get(1) {
                eprintln!("{DIM}delegating to @{target}...{RESET}");
                match kernel.delegate(current_pid, target, command).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                }
            } else {
                if let Some(proc) = kernel.find_by_name(target) {
                    current_pid = proc.pid;
                    eprintln!("{GREEN}Switched to @{target} (pid:{current_pid}){RESET}");
                } else {
                    eprintln!("{RED}NPC '{target}' not found.{RESET} Available:");
                    for p in kernel.ps() {
                        eprintln!("  {CYAN}@{}{RESET}", p.npc.name);
                    }
                }
            }
            continue;
        }

        if input.starts_with('/') {
            let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
            let cmd_name = parts[0];
            let args_str = parts.get(1).unwrap_or(&"");

            if kernel.jinxes.contains_key(cmd_name) {
                let mut args = std::collections::HashMap::new();

                if !args_str.is_empty() {
                    let mut has_kv = false;
                    for part in args_str.split_whitespace() {
                        if let Some((k, v)) = part.split_once('=') {
                            args.insert(k.to_string(), v.to_string());
                            has_kv = true;
                        }
                    }
                    if !has_kv {
                        if let Some(first_input) = kernel.jinxes[cmd_name].inputs.first() {
                            args.insert(first_input.name.clone(), args_str.to_string());
                        }
                    }
                }

                match kernel.syscall(current_pid, cmd_name, &args).await {
                    Ok(output) => {
                        if !output.is_empty() {
                            println!("{}", output);
                        }
                    }
                    Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                }
            } else {
                eprintln!("{RED}Unknown command: /{cmd_name}{RESET}");
            }
            continue;
        }

        if input.starts_with("cd ") || input == "cd" {
            let target = input.strip_prefix("cd").unwrap().trim();
            let target = if target.is_empty() {
                shellexpand::tilde("~").to_string()
            } else {
                shellexpand::tilde(target).to_string()
            };
            let target = if std::path::Path::new(&target).is_relative() {
                let cwd = std::env::current_dir().unwrap_or_default();
                cwd.join(&target)
                    .canonicalize()
                    .unwrap_or_else(|_| cwd.join(&target))
                    .display()
                    .to_string()
            } else {
                target
            };
            match std::env::set_current_dir(&target) {
                Ok(_) => eprintln!("{DIM}Changed to: {target}{RESET}"),
                Err(e) => eprintln!("{RED}cd: {e}{RESET}"),
            }
            continue;
        }

        if is_terminal_editor(&input) {
            run_interactive(&input);
            continue;
        }

        if is_interactive(&input) {
            run_interactive(&input);
            continue;
        }

        turn_count += 1;

        match mode {
            Mode::Agent => {
                if is_bash_command(&input) {
                    run_bash(&input).await;
                } else {
                    match kernel.exec(current_pid, &input).await {
                        Ok(output) if !output.is_empty() => println!("{}", output),
                        Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                        _ => {}
                    }
                }
            }
            Mode::Chat => {
                match kernel.exec_chat(current_pid, &input).await {
                    Ok(output) if !output.is_empty() => println!("{}", output),
                    Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                    _ => {}
                }
            }
            Mode::Cmd => {
                if !run_bash(&input).await {
                    match kernel.exec(current_pid, &input).await {
                        Ok(output) if !output.is_empty() => println!("{}", output),
                        Err(e) => eprintln!("{RED}Error: {e}{RESET}"),
                        _ => {}
                    }
                }
            }
        }

        if let Some(p) = kernel.get_process(current_pid) {
            if p.usage.total_turns > 0 {
                eprintln!(
                    "{DIM}[tokens:{}/{} | turn:{} | cost:${:.4}]{RESET}",
                    p.usage.total_input_tokens,
                    p.usage.total_output_tokens,
                    p.usage.total_turns,
                    p.usage.total_cost_usd,
                );
            }
        }
    }

    let _ = rl.save_history(&history_path);

    eprintln!("\n{DIM}Kernel shutting down.{RESET}");
    let s = kernel.stats();
    eprintln!(
        "{DIM}uptime: {}s | tokens: {} | cost: ${:.4}{RESET}",
        s.uptime_secs, s.total_tokens, s.total_cost_usd
    );
    Ok(())
}

fn handle_set_command(rest: &str, kernel: &mut Kernel, pid: u32, mode: &mut Mode) {
    let parts: Vec<&str> = rest.splitn(2, '=').collect();
    if parts.len() != 2 {
        eprintln!("Usage: /set key=value");
        eprintln!("  model=gpt-4o  provider=openai  mode=chat");
        return;
    }
    let key = parts[0].trim();
    let value = parts[1].trim();

    match key {
        "model" => {
            if let Some(p) = kernel.get_process_mut(pid) {
                p.npc.model = Some(value.to_string());
                eprintln!("{GREEN}model = {value}{RESET}");
            }
        }
        "provider" => {
            if let Some(p) = kernel.get_process_mut(pid) {
                p.npc.provider = Some(value.to_string());
                eprintln!("{GREEN}provider = {value}{RESET}");
            }
        }
        "mode" => match value {
            "agent" => *mode = Mode::Agent,
            "chat" => *mode = Mode::Chat,
            "cmd" => *mode = Mode::Cmd,
            _ => eprintln!("{RED}Unknown mode: {value}{RESET}"),
        },
        _ => eprintln!("{RED}Unknown setting: {key}{RESET}"),
    }
}

fn print_welcome(kernel: &Kernel) {
    let s = kernel.stats();

    eprintln!("{BOLD}{PURPLE}");
    eprintln!("  ███╗   ██╗██████╗  ██████╗███████╗██╗  ██╗");
    eprintln!("  ████╗  ██║██╔══██╗██╔════╝██╔════╝██║  ██║");
    eprintln!("  ██╔██╗ ██║██████╔╝██║     ███████╗███████║");
    eprintln!("  ██║╚██╗██║██╔═══╝ ██║     ╚════██║██╔══██║");
    eprintln!("  ██║ ╚████║██║     ╚██████╗███████║██║  ██║");
    eprintln!("  ╚═╝  ╚═══╝╚═╝      ╚═════╝╚══════╝╚═╝  ╚═╝");
    eprintln!("{RESET}");
    eprintln!("  {BOLD}npcsh-rs{RESET} v{} — NPC Operating System Shell", env!("CARGO_PKG_VERSION"));
    eprintln!("  {DIM}{} processes | {} jinxes | /help for commands{RESET}", s.total_processes, s.jinx_count);
    eprintln!();

    eprint!("  {BOLD}Agents:{RESET} ");
    let names: Vec<String> = kernel.ps().iter().map(|p| format!("{CYAN}@{}{RESET}", p.npc.name)).collect();
    eprintln!("{}", names.join("  "));

    eprintln!("  {BOLD}Modes:{RESET}  {CYAN}/agent{RESET}  {CYAN}/chat{RESET}  {CYAN}/cmd{RESET}");
    eprintln!();
}

fn load_npcshrc() {
    let rc_path = shellexpand::tilde("~/.npcshrc").to_string();
    let path = std::path::Path::new(&rc_path);

    if !path.exists() {
        return;
    }

    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line = line.strip_prefix("export ").unwrap_or(line);

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"').trim_matches('\'');
                if std::env::var(key).is_err() {
                    unsafe { std::env::set_var(key, value) };
                }
            }
        }
    }
}

const TERMINAL_EDITORS: &[&str] = &[
    "vim", "nvim", "nano", "vi", "emacs", "less", "more", "man",
];

const INTERACTIVE_COMMANDS: &[&str] = &[
    "ipython", "python", "python3", "node", "irb", "ghci",
    "mysql", "psql", "sqlite3", "redis-cli", "mongo",
    "ssh", "telnet", "ftp", "sftp", "top", "htop", "watch", "r",
];

const SHELL_BUILTINS: &[&str] = &[
    "cd", "pwd", "echo", "export", "source", "alias", "unalias",
    "history", "set", "unset", "read", "eval", "exec", "exit",
    "return", "shift", "trap", "wait", "jobs", "fg", "bg",
    "kill", "ulimit", "umask", "type", "hash", "true", "false",
];

fn is_bash_command(input: &str) -> bool {
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }

    let cmd = parts[0];

    if SHELL_BUILTINS.contains(&cmd) {
        return true;
    }

    if let Ok(output) = std::process::Command::new("which")
        .arg(cmd)
        .output()
    {
        return output.status.success();
    }

    false
}

fn is_terminal_editor(input: &str) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    TERMINAL_EDITORS.contains(&cmd)
}

fn is_interactive(input: &str) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    INTERACTIVE_COMMANDS.contains(&cmd)
}

async fn run_bash(input: &str) -> bool {
    match tokio::process::Command::new("bash")
        .arg("-c")
        .arg(input)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .await
    {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("{RED}bash: {e}{RESET}");
            false
        }
    }
}

fn run_interactive(input: &str) {
    let _ = std::process::Command::new("bash")
        .arg("-c")
        .arg(input)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();
}

fn find_team_dir() -> String {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--team") {
        if let Some(dir) = args.get(pos + 1) {
            return dir.clone();
        }
    }

    if std::path::Path::new("./npc_team").exists() {
        return "./npc_team".to_string();
    }

    let global = shellexpand::tilde("~/.npcsh/npc_team").to_string();
    if std::path::Path::new(&global).exists() {
        return global;
    }

    ".".to_string()
}
