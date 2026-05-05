use crate::index::Dataset;

pub const K: usize = 4096;
pub const NPROBE_PRIMARY: usize = 8;
pub const NPROBE_REFINE: usize = 24;
pub const BLOCK_SIZE: usize = 16;
pub const DIM: usize = 14;
pub const QUANT_SCALE: f32 = 10_000.0;

pub struct Top5 {
    dist: [u64; 5],
    label: [u8; 5],
    id: [u32; 5],
    worst_pos: usize,
}

impl Top5 {
    pub fn new() -> Self {
        Self {
            dist: [u64::MAX; 5],
            label: [0u8; 5],
            id: [u32::MAX; 5],
            worst_pos: 0,
        }
    }

    #[inline]
    pub fn worst(&self) -> u64 {
        self.dist[self.worst_pos]
    }

    #[inline]
    pub fn try_insert(&mut self, d: u64, lbl: u8, id: u32) {
        if d >= self.dist[self.worst_pos] {
            return;
        }
        self.dist[self.worst_pos] = d;
        self.label[self.worst_pos] = lbl;
        self.id[self.worst_pos] = id;
        let mut wp = 0;
        for i in 1..5 {
            if self.dist[i] > self.dist[wp] {
                wp = i;
            }
        }
        self.worst_pos = wp;
    }

    pub fn labels(&self) -> [u8; 5] {
        self.label
    }
}

#[inline]
pub fn distance_sq(a: &[i16; DIM], b: &[i16; DIM]) -> u64 {
    let mut sum = 0i64;
    for i in 0..DIM {
        let d = a[i] as i32 - b[i] as i32;
        sum += (d * d) as i64;
    }
    sum as u64
}

pub fn count_frauds(labels: &[u8; 5]) -> usize {
    labels.iter().filter(|&&l| l == 1).count()
}

pub fn search_top5_brute(q: &[i16; DIM], ds: &Dataset) -> [u8; 5] {
    let mut top = Top5::new();
    for i in 0..ds.n_vectors() {
        let v = ds.vector(i);
        let d = distance_sq(q, &v);
        top.try_insert(d, ds.label(i), i as u32);
    }
    top.labels()
}

pub fn search_top5(q: &[i16; DIM], ds: &Dataset) -> [u8; 5] {
    let centroid_dists = compute_centroid_dists(q, &ds.centroids);
    let ranked = rank_centroids(&centroid_dists, NPROBE_REFINE);

    let mut top = Top5::new();
    for slot in 0..NPROBE_PRIMARY {
        let ci = ranked[slot].1 as usize;
        scan_cluster(q, ds, ci, &mut top);
    }

    let primary_frauds = count_frauds(&top.labels());
    if matches!(primary_frauds, 2 | 3) {
        for slot in NPROBE_PRIMARY..NPROBE_REFINE {
            let ci = ranked[slot].1 as usize;
            scan_cluster(q, ds, ci, &mut top);
        }
    }

    top.labels()
}

fn rank_centroids(dists: &[f32], take: usize) -> Vec<(f32, u32)> {
    let take = take.min(dists.len());
    let mut out: Vec<(f32, u32)> = (0..take).map(|i| (dists[i], i as u32)).collect();
    out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut worst_pos = take.saturating_sub(1);
    for ci in take..dists.len() {
        if dists[ci] < out[worst_pos].0 {
            out[worst_pos] = (dists[ci], ci as u32);
            let mut wp = 0;
            for i in 1..take {
                if out[i].0 > out[wp].0 {
                    wp = i;
                }
            }
            worst_pos = wp;
        }
    }
    out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn compute_centroid_dists(q: &[i16; DIM], centroids: &[f32]) -> Vec<f32> {
    let q_f32: [f32; DIM] = std::array::from_fn(|i| q[i] as f32 / QUANT_SCALE);

    let mut dists = vec![0f32; K];

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        compute_centroid_dists_avx2(&q_f32, centroids, &mut dists);
        return dists;
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        for ci in 0..K {
            let mut s = 0f32;
            for d in 0..DIM {
                let cv = centroids[d * K + ci];
                let diff = cv - q_f32[d];
                s += diff * diff;
            }
            dists[ci] = s;
        }
        dists
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_centroid_dists_avx2(q: &[f32; DIM], centroids: &[f32], dists: &mut [f32]) {
    use std::arch::x86_64::*;
    let cp = centroids.as_ptr();
    let dp = dists.as_mut_ptr();

    let qd0 = _mm256_set1_ps(q[0]);
    let mut ci = 0usize;
    while ci + 16 <= K {
        let c0 = _mm256_loadu_ps(cp.add(ci));
        let c1 = _mm256_loadu_ps(cp.add(ci + 8));
        let d0 = _mm256_sub_ps(c0, qd0);
        let d1 = _mm256_sub_ps(c1, qd0);
        _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
        _mm256_storeu_ps(dp.add(ci + 8), _mm256_mul_ps(d1, d1));
        ci += 16;
    }
    while ci + 8 <= K {
        let c0 = _mm256_loadu_ps(cp.add(ci));
        let d0 = _mm256_sub_ps(c0, qd0);
        _mm256_storeu_ps(dp.add(ci), _mm256_mul_ps(d0, d0));
        ci += 8;
    }
    while ci < K {
        let diff = *cp.add(ci) - q[0];
        *dp.add(ci) = diff * diff;
        ci += 1;
    }

    for d in 1..DIM {
        let row = d * K;
        let qd = _mm256_set1_ps(q[d]);
        let mut ci = 0usize;
        while ci + 16 <= K {
            let c0 = _mm256_loadu_ps(cp.add(row + ci));
            let c1 = _mm256_loadu_ps(cp.add(row + ci + 8));
            let diff0 = _mm256_sub_ps(c0, qd);
            let diff1 = _mm256_sub_ps(c1, qd);
            let a0 = _mm256_loadu_ps(dp.add(ci));
            let a1 = _mm256_loadu_ps(dp.add(ci + 8));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(diff0, diff0, a0));
            _mm256_storeu_ps(dp.add(ci + 8), _mm256_fmadd_ps(diff1, diff1, a1));
            ci += 16;
        }
        while ci + 8 <= K {
            let cv = _mm256_loadu_ps(cp.add(row + ci));
            let diff = _mm256_sub_ps(cv, qd);
            let a = _mm256_loadu_ps(dp.add(ci));
            _mm256_storeu_ps(dp.add(ci), _mm256_fmadd_ps(diff, diff, a));
            ci += 8;
        }
        while ci < K {
            let diff = *cp.add(row + ci) - q[d];
            *dp.add(ci) += diff * diff;
            ci += 1;
        }
    }
}

fn scan_cluster(q: &[i16; DIM], ds: &Dataset, ci: usize, top: &mut Top5) {
    let start = ds.offsets[ci] as usize;
    let end = ds.offsets[ci + 1] as usize;
    if end <= start {
        return;
    }

    let first_block = start / BLOCK_SIZE;
    let last_block_excl = (end + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        for blk in first_block..last_block_excl {
            unsafe { scan_block16_avx2(q, ds, blk, start, end, top) };
        }
        return;
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for blk in first_block..last_block_excl {
            scan_block16_scalar(q, ds, blk, start, end, top);
        }
    }
}

fn scan_block16_scalar(
    q: &[i16; DIM],
    ds: &Dataset,
    blk: usize,
    cluster_start: usize,
    cluster_end: usize,
    top: &mut Top5,
) {
    let block_base = blk * BLOCK_SIZE * DIM;
    let lane_start = blk * BLOCK_SIZE;
    for lane in 0..BLOCK_SIZE {
        let g = lane_start + lane;
        if g < cluster_start || g >= cluster_end {
            continue;
        }
        let mut s = 0i64;
        for d in 0..DIM {
            let v = ds.blocks[block_base + d * BLOCK_SIZE + lane] as i32;
            let qd = q[d] as i32;
            let diff = v - qd;
            s += (diff * diff) as i64;
        }
        top.try_insert(s as u64, ds.labels[g], ds.orig_ids[g]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn scan_block16_avx2(
    q: &[i16; DIM],
    ds: &Dataset,
    blk: usize,
    cluster_start: usize,
    cluster_end: usize,
    top: &mut Top5,
) {
    use std::arch::x86_64::*;

    let block_base = blk * BLOCK_SIZE * DIM;
    let lane_start = blk * BLOCK_SIZE;
    let block_ptr = ds.blocks.as_ptr().add(block_base);

    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();

    for d in 0..DIM {
        let qd = _mm256_set1_epi32(q[d] as i32);
        let dim_ptr = block_ptr.add(d * BLOCK_SIZE);
        let v_lo_i16 = _mm_loadu_si128(dim_ptr as *const _);
        let v_hi_i16 = _mm_loadu_si128(dim_ptr.add(8) as *const _);
        let v_lo = _mm256_cvtepi16_epi32(v_lo_i16);
        let v_hi = _mm256_cvtepi16_epi32(v_hi_i16);
        let diff_lo = _mm256_sub_epi32(v_lo, qd);
        let diff_hi = _mm256_sub_epi32(v_hi, qd);
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_mullo_epi32(diff_lo, diff_lo));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_mullo_epi32(diff_hi, diff_hi));
    }

    let mut tmp = [0i32; 16];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, acc_lo);
    _mm256_storeu_si256(tmp.as_mut_ptr().add(8) as *mut _, acc_hi);

    for lane in 0..BLOCK_SIZE {
        let g = lane_start + lane;
        if g < cluster_start || g >= cluster_end {
            continue;
        }
        let d = tmp[lane] as u64;
        top.try_insert(d, ds.labels[g], ds.orig_ids[g]);
    }
}

fn bbox_lower_bound(q: &[i16; DIM], ds: &Dataset, ci: usize) -> u64 {
    let mut s = 0u64;
    let base = ci * DIM;
    for d in 0..DIM {
        let mn = ds.bbox_min[base + d] as i32;
        let mx = ds.bbox_max[base + d] as i32;
        let qd = q[d] as i32;
        let dd = if qd < mn {
            (mn - qd) as i64
        } else if qd > mx {
            (qd - mx) as i64
        } else {
            0
        };
        s += (dd * dd) as u64;
    }
    s
}

pub fn warmup(n_queries: usize, ds: &Dataset) {
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..n_queries {
        let mut q = [0i16; DIM];
        for d in 0..DIM {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            q[d] = ((seed >> 48) as i16).wrapping_rem(20_001).wrapping_sub(10_000);
        }
        let _ = search_top5(&q, ds);
    }
}
