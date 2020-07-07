#[cfg(target_arch = "x86_64")]
fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    if target_arch == "x86_64" {
        cc::Build::new()
            .flag("-c")
            .file("./asm/ext_twisted_ed_add.S")
            .compile("libsncrypto-twisted.a");
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {}
