use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};

use flate2::read::GzDecoder;
use serde::Deserialize;

#[path = "../index.rs"]
mod index;
#[path = "../ivf.rs"]
mod ivf;
#[path = "../vectorizer.rs"]
mod vectorizer;

use ivf::DIM;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; DIM],
    label: String,
}

#[derive(Deserialize)]
struct TestData {
    entries: Vec<TestQuery>,
}

#[derive(Deserialize)]
struct TestQuery {
    request: serde_json::Value,
    expected_approved: bool,
    #[serde(default)]
    #[allow(dead_code)]
    expected_fraud_score: f32,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: verify <index.bin> <references.json.gz> <test-data.json> [n_queries]"
        );
        std::process::exit(2);
    }
    let index_path = &args[1];
    let refs_path = &args[2];
    let test_path = &args[3];
    let n_queries: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    eprintln!("[verify] loading index {}", index_path);
    index::load_mmap(index_path)?;
    let ds = index::get();

    eprintln!("[verify] loading references {}", refs_path);
    let (raw_vectors, raw_labels) = read_references(refs_path)?;
    eprintln!("[verify] loaded {} references", raw_vectors.len());

    eprintln!("[verify] loading test queries {}", test_path);
    let td: TestData = serde_json::from_reader(BufReader::new(File::open(test_path)?))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let test_data = td.entries;
    eprintln!("[verify] loaded {} test queries", test_data.len());

    let mcc = vectorizer::MccTable::load("resources/mcc_risk.json")?;

    let n = n_queries.min(test_data.len());
    let mut ivf_correct = 0usize;
    let mut brute_correct = 0usize;
    let mut top5_match = 0usize;
    let mut decision_match = 0usize;

    for q_idx in 0..n {
        let payload: vectorizer::Payload = serde_json::from_value(test_data[q_idx].request.clone())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let q = vectorizer::vectorize(&payload, &mcc);

        let ivf_labels = ivf::search_top5(&q, ds);
        let ivf_frauds = ivf::count_frauds(&ivf_labels);
        let ivf_score = ivf_frauds as f32 / 5.0;
        let ivf_approved = ivf_score < 0.6;

        let brute_top5 = brute_force_top5(&q, &raw_vectors, &raw_labels);
        let brute_frauds = brute_top5.iter().filter(|&&l| l == 1).count();
        let brute_score = brute_frauds as f32 / 5.0;
        let brute_approved = brute_score < 0.6;

        let exp_approved = test_data[q_idx].expected_approved;
        if ivf_approved == exp_approved {
            ivf_correct += 1;
        }
        if brute_approved == exp_approved {
            brute_correct += 1;
        }
        if ivf_frauds == brute_frauds {
            top5_match += 1;
        }
        if ivf_approved == brute_approved {
            decision_match += 1;
        }
    }

    println!("queries: {}", n);
    println!(
        "  IVF vs expected      : {}/{} ({:.2}%)",
        ivf_correct,
        n,
        100.0 * ivf_correct as f32 / n as f32
    );
    println!(
        "  brute vs expected    : {}/{} ({:.2}%)",
        brute_correct,
        n,
        100.0 * brute_correct as f32 / n as f32
    );
    println!(
        "  IVF == brute (frauds): {}/{} ({:.2}%)",
        top5_match,
        n,
        100.0 * top5_match as f32 / n as f32
    );
    println!(
        "  IVF == brute (approved): {}/{} ({:.2}%)",
        decision_match,
        n,
        100.0 * decision_match as f32 / n as f32
    );

    Ok(())
}

fn brute_force_top5(q: &[i16; DIM], refs: &[[i16; DIM]], labels: &[u8]) -> [u8; 5] {
    use rayon::prelude::*;

    #[derive(Clone, Copy)]
    struct Cand {
        d: u64,
        l: u8,
        i: u32,
    }

    let cands: Vec<Cand> = (0..refs.len())
        .into_par_iter()
        .map(|i| {
            let d = ivf::distance_sq(q, &refs[i]);
            Cand {
                d,
                l: labels[i],
                i: i as u32,
            }
        })
        .collect();

    let mut top: [Cand; 5] = [Cand { d: u64::MAX, l: 0, i: u32::MAX }; 5];
    let mut worst_pos = 0;
    for c in &cands {
        if c.d >= top[worst_pos].d {
            continue;
        }
        top[worst_pos] = *c;
        let mut wp = 0;
        for i in 1..5 {
            if top[i].d > top[wp].d {
                wp = i;
            }
        }
        worst_pos = wp;
    }

    let mut out = [0u8; 5];
    for i in 0..5 {
        out[i] = top[i].l;
    }
    out
}

fn read_references(path: &str) -> io::Result<(Vec<[i16; DIM]>, Vec<u8>)> {
    let f = File::open(path)?;
    let buf = BufReader::new(f);
    let mut reader: Box<dyn Read> = if path.ends_with(".gz") {
        Box::new(GzDecoder::new(buf))
    } else {
        Box::new(buf)
    };
    let mut all = String::new();
    reader.read_to_string(&mut all)?;
    let parsed: Vec<Reference> = serde_json::from_str(&all)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let n = parsed.len();
    let mut vectors_q = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for r in parsed {
        let q = vectorizer::quantize(&r.vector);
        vectors_q.push(q);
        labels.push(if r.label == "fraud" { 1u8 } else { 0u8 });
    }
    Ok((vectors_q, labels))
}
