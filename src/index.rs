use bytemuck::{cast_slice, Pod, Zeroable};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;

use crate::ivf::{BLOCK_SIZE, DIM, K};

pub const MAGIC: [u8; 4] = *b"RIVF";
pub const VERSION: u32 = 1;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub n_vectors: u32,
    pub k: u32,
    pub dim: u32,
    pub block_size: u32,
    pub n_blocks: u32,
    pub _pad: u32,
}

pub struct Dataset {
    _mmap: Option<Mmap>,
    pub n_vectors: usize,
    pub n_blocks: usize,
    pub centroids: Vec<f32>,
    pub bbox_min: Vec<i16>,
    pub bbox_max: Vec<i16>,
    pub offsets: Vec<u32>,
    pub blocks: Vec<i16>,
    pub labels: Vec<u8>,
    pub orig_ids: Vec<u32>,
}

static DATASET: OnceLock<Dataset> = OnceLock::new();

pub fn get() -> &'static Dataset {
    DATASET.get().expect("dataset not loaded")
}

pub fn load_mmap(path: impl AsRef<Path>) -> io::Result<()> {
    let file = File::open(path.as_ref())?;
    let mmap = unsafe { Mmap::map(&file)? };
    let ds = parse_mmap(mmap)?;
    DATASET
        .set(ds)
        .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "dataset already loaded"))?;
    Ok(())
}

fn parse_mmap(mmap: Mmap) -> io::Result<Dataset> {
    let bytes = &mmap[..];
    if bytes.len() < std::mem::size_of::<Header>() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
    }
    let header: Header = *bytemuck::from_bytes(&bytes[..std::mem::size_of::<Header>()]);
    if header.magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    if header.version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad version"));
    }
    if header.k as usize != K
        || header.dim as usize != DIM
        || header.block_size as usize != BLOCK_SIZE
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "param mismatch: k={} dim={} bs={}",
                header.k, header.dim, header.block_size
            ),
        ));
    }

    let n = header.n_vectors as usize;
    let nb = header.n_blocks as usize;
    let k = header.k as usize;

    let mut off = std::mem::size_of::<Header>();

    let centroids_bytes = k * DIM * 4;
    let centroids: Vec<f32> = cast_slice::<u8, f32>(&bytes[off..off + centroids_bytes]).to_vec();
    off += centroids_bytes;

    let bbox_bytes = k * DIM * 2;
    let bbox_min: Vec<i16> = cast_slice::<u8, i16>(&bytes[off..off + bbox_bytes]).to_vec();
    off += bbox_bytes;
    let bbox_max: Vec<i16> = cast_slice::<u8, i16>(&bytes[off..off + bbox_bytes]).to_vec();
    off += bbox_bytes;

    let offsets_bytes = (k + 1) * 4;
    let offsets: Vec<u32> = cast_slice::<u8, u32>(&bytes[off..off + offsets_bytes]).to_vec();
    off += offsets_bytes;

    let blocks_bytes = nb * BLOCK_SIZE * DIM * 2;
    let blocks: Vec<i16> = cast_slice::<u8, i16>(&bytes[off..off + blocks_bytes]).to_vec();
    off += blocks_bytes;

    let labels_padded = nb * BLOCK_SIZE;
    let labels: Vec<u8> = bytes[off..off + labels_padded].to_vec();
    off += labels_padded;

    let ids_bytes = labels_padded * 4;
    let orig_ids: Vec<u32> = cast_slice::<u8, u32>(&bytes[off..off + ids_bytes]).to_vec();

    Ok(Dataset {
        _mmap: Some(mmap),
        n_vectors: n,
        n_blocks: nb,
        centroids,
        bbox_min,
        bbox_max,
        offsets,
        blocks,
        labels,
        orig_ids,
    })
}

impl Dataset {
    pub fn n_vectors(&self) -> usize {
        self.n_vectors
    }

    pub fn vector(&self, global_idx: usize) -> [i16; DIM] {
        let block = global_idx / BLOCK_SIZE;
        let lane = global_idx % BLOCK_SIZE;
        let base = block * BLOCK_SIZE * DIM;
        let mut out = [0i16; DIM];
        for d in 0..DIM {
            out[d] = self.blocks[base + d * BLOCK_SIZE + lane];
        }
        out
    }

    pub fn label(&self, global_idx: usize) -> u8 {
        self.labels[global_idx]
    }
}

pub fn write_index(
    path: impl AsRef<Path>,
    n_vectors: usize,
    centroids: &[f32],
    bbox_min: &[i16],
    bbox_max: &[i16],
    offsets: &[u32],
    blocks: &[i16],
    labels: &[u8],
    orig_ids: &[u32],
) -> io::Result<()> {
    let n_blocks = (n_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let header = Header {
        magic: MAGIC,
        version: VERSION,
        n_vectors: n_vectors as u32,
        k: K as u32,
        dim: DIM as u32,
        block_size: BLOCK_SIZE as u32,
        n_blocks: n_blocks as u32,
        _pad: 0,
    };

    let mut f = File::create(path)?;
    f.write_all(bytemuck::bytes_of(&header))?;
    f.write_all(cast_slice::<f32, u8>(centroids))?;
    f.write_all(cast_slice::<i16, u8>(bbox_min))?;
    f.write_all(cast_slice::<i16, u8>(bbox_max))?;
    f.write_all(cast_slice::<u32, u8>(offsets))?;
    f.write_all(cast_slice::<i16, u8>(blocks))?;
    f.write_all(labels)?;
    f.write_all(cast_slice::<u32, u8>(orig_ids))?;
    Ok(())
}
