#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod index;
mod ivf;
mod parser;
mod vectorizer;

#[cfg(target_os = "linux")]
mod server;

#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    server::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("solution-x server requires linux (io_uring). Use Docker for local runs.");
    std::process::exit(1);
}
