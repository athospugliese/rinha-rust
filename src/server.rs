use std::env;
use std::io;
use std::rc::Rc;

use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use glommio::net::{UnixListener, UnixStream};
use glommio::LocalExecutor;

use crate::index;
use crate::ivf;
use crate::parser;
use crate::vectorizer::MccTable;

const RX_CAP: usize = 16 * 1024;

static RESP_FRAUD: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

static RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
static RESP_404: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
static RESP_400: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

pub fn run() -> io::Result<()> {
    let uds = env::var("UDS_PATH").unwrap_or_else(|_| "/sockets/api1.sock".into());
    let index_path = env::var("INDEX_PATH").unwrap_or_else(|_| "/data/index.bin".into());
    let mcc_path = env::var("MCC_PATH").unwrap_or_else(|_| "/data/mcc_risk.json".into());
    let warmup_n: usize = env::var("WARMUP_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    let _ = std::fs::remove_file(&uds);

    index::load_mmap(&index_path)?;
    let mcc = Rc::new(MccTable::load(&mcc_path)?);

    eprintln!("solution-x ready: uds={} index={}", uds, index_path);

    let ex = LocalExecutor::default();
    ex.run(async move {
        if warmup_n > 0 {
            ivf::warmup(warmup_n, index::get());
            eprintln!("warmup done ({} queries)", warmup_n);
        }
        let listener = match UnixListener::bind(&uds) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("bind error: {e}");
                return;
            }
        };
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&uds, std::fs::Permissions::from_mode(0o666));
        eprintln!("listening on {}", uds);
        loop {
            match listener.accept().await {
                Ok(stream) => {
                    let mcc = Rc::clone(&mcc);
                    glommio::spawn_local(handle(stream, mcc)).detach();
                }
                Err(e) => {
                    eprintln!("accept error: {e}");
                    break;
                }
            }
        }
    });

    Ok(())
}

async fn handle(mut stream: UnixStream, mcc: Rc<MccTable>) {
    let mut buf = vec![0u8; RX_CAP];
    let mut filled: usize = 0;
    let mut tx_buf: Vec<u8> = Vec::with_capacity(2048);
    let mut close_after = false;

    loop {
        if filled >= buf.len() {
            let _ = stream.write_all(RESP_400).await;
            return;
        }
        let n = match stream.read(&mut buf[filled..]).await {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        filled += n;
        tx_buf.clear();
        close_after = false;

        loop {
            match parse_request(&buf[..filled]) {
                ParseResult::Incomplete => break,
                ParseResult::NeedBody => break,
                ParseResult::Bad => {
                    tx_buf.extend_from_slice(RESP_400);
                    close_after = true;
                    break;
                }
                ParseResult::Ready { consumed } => {
                    tx_buf.extend_from_slice(RESP_READY);
                    consume(&mut buf, &mut filled, consumed);
                }
                ParseResult::NotFound { consumed } => {
                    tx_buf.extend_from_slice(RESP_404);
                    consume(&mut buf, &mut filled, consumed);
                    close_after = true;
                    break;
                }
                ParseResult::Fraud {
                    body_start,
                    body_end,
                    consumed,
                } => {
                    let body = &buf[body_start..body_end];
                    let resp = match parser::parse_and_vectorize(body, &mcc) {
                        Some(q) => {
                            let labels = ivf::search_top5(&q, index::get());
                            let frauds = ivf::count_frauds(&labels);
                            RESP_FRAUD[frauds.min(5)]
                        }
                        None => RESP_FRAUD[0],
                    };
                    tx_buf.extend_from_slice(resp);
                    consume(&mut buf, &mut filled, consumed);
                }
            }
        }

        if !tx_buf.is_empty() {
            if stream.write_all(&tx_buf).await.is_err() {
                return;
            }
        }
        if close_after {
            return;
        }
    }
}

fn consume(buf: &mut [u8], filled: &mut usize, consumed: usize) {
    if consumed >= *filled {
        *filled = 0;
    } else {
        buf.copy_within(consumed..*filled, 0);
        *filled -= consumed;
    }
}

enum ParseResult {
    Incomplete,
    NeedBody,
    Bad,
    Ready { consumed: usize },
    NotFound { consumed: usize },
    Fraud { body_start: usize, body_end: usize, consumed: usize },
}

fn parse_request(buf: &[u8]) -> ParseResult {
    let header_end = match find_double_crlf(buf) {
        Some(p) => p,
        None => return ParseResult::Incomplete,
    };
    let header_len = header_end + 4;

    if buf.len() < 4 {
        return ParseResult::Bad;
    }

    if starts_with(buf, b"GET /ready") {
        return ParseResult::Ready { consumed: header_len };
    }

    if starts_with(buf, b"POST /fraud-score") {
        let cl = match find_content_length(&buf[..header_end]) {
            Some(n) => n,
            None => return ParseResult::Bad,
        };
        if buf.len() < header_len + cl {
            return ParseResult::NeedBody;
        }
        return ParseResult::Fraud {
            body_start: header_len,
            body_end: header_len + cl,
            consumed: header_len + cl,
        };
    }

    ParseResult::NotFound { consumed: header_len }
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    for i in 0..=buf.len() - 4 {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' && buf[i + 2] == b'\r' && buf[i + 3] == b'\n' {
            return Some(i);
        }
    }
    None
}

fn starts_with(buf: &[u8], prefix: &[u8]) -> bool {
    buf.len() >= prefix.len() && &buf[..prefix.len()] == prefix
}

fn find_content_length(headers: &[u8]) -> Option<usize> {
    let key = b"content-length:";
    let mut i = 0;
    while i + key.len() <= headers.len() {
        let at_line_start = i == 0 || headers[i - 1] == b'\n';
        if at_line_start && headers[i..i + key.len()].eq_ignore_ascii_case(key) {
            let mut j = i + key.len();
            while j < headers.len() && (headers[j] == b' ' || headers[j] == b'\t') {
                j += 1;
            }
            let mut n: usize = 0;
            let mut any = false;
            while j < headers.len() && headers[j].is_ascii_digit() {
                n = n * 10 + (headers[j] - b'0') as usize;
                j += 1;
                any = true;
            }
            if any {
                return Some(n);
            }
            return None;
        }
        i += 1;
    }
    None
}
