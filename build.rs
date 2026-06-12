fn main() {
    // Try to read the Python npcsh VERSION file (sibling repo) first
    let version = std::fs::read_to_string("../npcsh/npcsh/VERSION")
        .or_else(|_| std::fs::read_to_string("../npcsh/VERSION"))
        .or_else(|_| std::fs::read_to_string("npcsh/VERSION"))
        .or_else(|_| std::fs::read_to_string("VERSION"))
        .unwrap_or_else(|_| "0.0.0".to_string())
        .trim()
        .to_string();

    println!("cargo:rustc-env=NPCSH_VERSION={}", version);
    println!("cargo:rustc-env=NPCRS_VERSION={}", version);
    println!("cargo:rerun-if-changed=VERSION");
    println!("cargo:rerun-if-changed=../npcsh/npcsh/VERSION");
}
