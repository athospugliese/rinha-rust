use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use flate2::read::GzDecoder;
use serde::Deserialize;

#[path = "../index.rs"]
mod index;
#[path = "../ivf.rs"]
mod ivf;
#[path = "../vectorizer.rs"]
mod vectorizer;

use ivf::{BLOCK_SIZE, DIM, K, QUANT_SCALE};

#[derive(Deserialize)]
struct Reference {
    vector: [f32; DIM],
    label: String,
}

const KMEANS_ITERATIONS: usize = 12;
const KMEANS_SAMPLE: usize = 65_536;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: build_index <references.json.gz> <out.index.bin>"
        );
        std::process::exit(2);
    }
    let in_path = &args[1];
    let out_path = &args[2];

    eprintln!("[build_index] reading {}", in_path);
    let (vectors_q, labels_raw) = read_references(in_path)?;
    let n = vectors_q.len();
    eprintln!("[build_index] loaded {} vectors", n);

    eprintln!(
        "[build_index] kmeans++ K={} sample={} iters={}",
        K, KMEANS_SAMPLE, KMEANS_ITERATIONS
    );
    let centroids = kmeans(&vectors_q, K, KMEANS_SAMPLE, KMEANS_ITERATIONS);
    eprintln!("[build_index] kmeans done");

    eprintln!("[build_index] assigning {} vectors to clusters (parallel)", n);
    use rayon::prelude::*;
    let assign: Vec<u32> = vectors_q
        .par_iter()
        .map(|v| nearest_centroid(v, &centroids) as u32)
        .collect();

    let mut counts = vec![0u32; K];
    for &c in &assign {
        counts[c as usize] += 1;
    }
    let max_c = counts.iter().copied().max().unwrap_or(0);
    let min_c = counts.iter().copied().min().unwrap_or(0);
    eprintln!(
        "[build_index] cluster sizes: min={} max={} avg={}",
        min_c,
        max_c,
        n / K
    );

    let mut offsets = vec![0u32; K + 1];
    for c in 0..K {
        offsets[c + 1] = offsets[c] + counts[c];
    }
    let total_n = offsets[K] as usize;
    assert_eq!(total_n, n);

    let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let padded_n = n_blocks * BLOCK_SIZE;

    let mut blocks = vec![0i16; n_blocks * BLOCK_SIZE * DIM];
    let mut labels = vec![0u8; padded_n];
    let mut orig_ids = vec![0u32; padded_n];

    let mut cursor = offsets.clone();
    for i in 0..n {
        let c = assign[i] as usize;
        let pos = cursor[c] as usize;
        cursor[c] += 1;
        let block = pos / BLOCK_SIZE;
        let lane = pos % BLOCK_SIZE;
        let block_base = block * BLOCK_SIZE * DIM;
        for d in 0..DIM {
            blocks[block_base + d * BLOCK_SIZE + lane] = vectors_q[i][d];
        }
        labels[pos] = labels_raw[i];
        orig_ids[pos] = i as u32;
    }

    eprintln!("[build_index] computing bbox per cluster");
    let mut bbox_min = vec![i16::MAX; K * DIM];
    let mut bbox_max = vec![i16::MIN; K * DIM];
    for c in 0..K {
        let s = offsets[c] as usize;
        let e = offsets[c + 1] as usize;
        for pos in s..e {
            let block = pos / BLOCK_SIZE;
            let lane = pos % BLOCK_SIZE;
            let block_base = block * BLOCK_SIZE * DIM;
            for d in 0..DIM {
                let v = blocks[block_base + d * BLOCK_SIZE + lane];
                let mb = c * DIM + d;
                if v < bbox_min[mb] {
                    bbox_min[mb] = v;
                }
                if v > bbox_max[mb] {
                    bbox_max[mb] = v;
                }
            }
        }
    }

    let mut centroids_soa = vec![0f32; K * DIM];
    for d in 0..DIM {
        for c in 0..K {
            centroids_soa[d * K + c] = centroids[c * DIM + d];
        }
    }

    eprintln!("[build_index] writing {}", out_path);
    index::write_index(
        Path::new(out_path),
        n,
        &centroids_soa,
        &bbox_min,
        &bbox_max,
        &offsets,
        &blocks,
        &labels,
        &orig_ids,
    )?;
    eprintln!("[build_index] done");
    Ok(())
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

fn kmeans(vectors: &[[i16; DIM]], k: usize, sample: usize, iters: usize) -> Vec<f32> {
    let n = vectors.len();
    let mut rng_state: u64 = 0x6E_4C9F_2A_71B0_5E1D;

    let sample = sample.min(n);
    let mut sample_idx = Vec::with_capacity(sample);
    for _ in 0..sample {
        rng_state = lcg(rng_state);
        sample_idx.push((rng_state as usize) % n);
    }

    let mut centroids = vec![0f32; k * DIM];
    rng_state = lcg(rng_state);
    let first = sample_idx[(rng_state as usize) % sample];
    for d in 0..DIM {
        centroids[d] = vectors[first][d] as f32 / QUANT_SCALE;
    }

    let mut chosen = 1usize;
    let mut min_d = vec![f32::MAX; sample];
    while chosen < k {
        for s in 0..sample {
            let i = sample_idx[s];
            let mut dist = 0f32;
            for d in 0..DIM {
                let v = vectors[i][d] as f32 / QUANT_SCALE;
                let c = centroids[(chosen - 1) * DIM + d];
                let dd = v - c;
                dist += dd * dd;
            }
            if dist < min_d[s] {
                min_d[s] = dist;
            }
        }
        let mut total = 0f64;
        for &d in &min_d {
            total += d as f64;
        }
        rng_state = lcg(rng_state);
        let r = (rng_state as f64 / u64::MAX as f64) * total;
        let mut acc = 0f64;
        let mut pick = sample - 1;
        for s in 0..sample {
            acc += min_d[s] as f64;
            if acc >= r {
                pick = s;
                break;
            }
        }
        let i = sample_idx[pick];
        for d in 0..DIM {
            centroids[chosen * DIM + d] = vectors[i][d] as f32 / QUANT_SCALE;
        }
        chosen += 1;
        if chosen % 128 == 0 {
            eprintln!("[build_index] kmeans++ init {}/{}", chosen, k);
        }
    }

    use rayon::prelude::*;
    for iter in 0..iters {
        let assign: Vec<u32> = vectors
            .par_iter()
            .map(|v| nearest_centroid(v, &centroids) as u32)
            .collect();

        let mut sums = vec![0f64; k * DIM];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            for d in 0..DIM {
                sums[c * DIM + d] += vectors[i][d] as f64 / QUANT_SCALE as f64;
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let inv = 1.0f64 / counts[c] as f64;
            for d in 0..DIM {
                centroids[c * DIM + d] = (sums[c * DIM + d] * inv) as f32;
            }
        }
        eprintln!("[build_index] kmeans iter {}/{}", iter + 1, iters);
    }
    centroids
}

fn nearest_centroid(v: &[i16; DIM], centroids: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    let kc = centroids.len() / DIM;
    for c in 0..kc {
        let mut s = 0f32;
        for d in 0..DIM {
            let cv = centroids[c * DIM + d];
            let qv = v[d] as f32 / QUANT_SCALE;
            let diff = cv - qv;
            s += diff * diff;
        }
        if s < best_d {
            best_d = s;
            best = c;
        }
    }
    best
}

fn lcg(s: u64) -> u64 {
    s.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
