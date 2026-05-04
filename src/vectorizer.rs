use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;

pub const DIM: usize = 14;
pub const QUANT_SCALE: f32 = 10_000.0;

#[derive(Deserialize)]
pub struct Payload {
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTx>,
}

#[derive(Deserialize)]
pub struct Transaction {
    pub amount: f32,
    pub installments: u32,
    pub requested_at: String,
}

#[derive(Deserialize)]
pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Deserialize)]
pub struct LastTx {
    pub timestamp: String,
    pub km_from_current: f32,
}

pub struct MccTable {
    map: HashMap<String, f32>,
}

impl MccTable {
    pub fn load(path: &str) -> io::Result<Self> {
        let content = fs::read_to_string(path)?;
        let map: HashMap<String, f32> = serde_json::from_str(&content)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self { map })
    }

    pub fn lookup(&self, mcc: &str) -> f32 {
        self.map.get(mcc).copied().unwrap_or(0.5)
    }

    pub fn lookup_bytes(&self, mcc: &[u8]) -> f32 {
        match std::str::from_utf8(mcc) {
            Ok(s) => self.lookup(s),
            Err(_) => 0.5,
        }
    }
}

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn vectorize_f32(p: &Payload, mcc: &MccTable) -> [f32; DIM] {
    let mut v = [0f32; DIM];
    v[0] = clamp01(p.transaction.amount / 10_000.0);
    v[1] = clamp01(p.transaction.installments as f32 / 12.0);
    v[2] = clamp01((p.transaction.amount / p.customer.avg_amount) / 10.0);

    let (hour, weekday, ts_minutes) = parse_iso(&p.transaction.requested_at);
    v[3] = hour as f32 / 23.0;
    v[4] = weekday as f32 / 6.0;

    match &p.last_transaction {
        Some(lt) => {
            let (_, _, prev_minutes) = parse_iso(&lt.timestamp);
            let delta = (ts_minutes as i64 - prev_minutes as i64).unsigned_abs() as f32;
            v[5] = clamp01(delta / 1440.0);
            v[6] = clamp01(lt.km_from_current / 1000.0);
        }
        None => {
            v[5] = -1.0;
            v[6] = -1.0;
        }
    }

    v[7] = clamp01(p.terminal.km_from_home / 1000.0);
    v[8] = clamp01(p.customer.tx_count_24h as f32 / 20.0);
    v[9] = if p.terminal.is_online { 1.0 } else { 0.0 };
    v[10] = if p.terminal.card_present { 1.0 } else { 0.0 };
    v[11] = if p
        .customer
        .known_merchants
        .iter()
        .any(|m| m == &p.merchant.id)
    {
        0.0
    } else {
        1.0
    };
    v[12] = mcc.lookup(&p.merchant.mcc);
    v[13] = clamp01(p.merchant.avg_amount / 10_000.0);
    v
}

pub fn quantize(v: &[f32; DIM]) -> [i16; DIM] {
    let mut out = [0i16; DIM];
    for i in 0..DIM {
        let s = v[i].clamp(-1.0, 1.0) * QUANT_SCALE;
        out[i] = s.round() as i16;
    }
    out
}

pub fn vectorize(p: &Payload, mcc: &MccTable) -> [i16; DIM] {
    quantize(&vectorize_f32(p, mcc))
}

fn parse_iso(s: &str) -> (u8, u8, i64) {
    let b = s.as_bytes();
    let year = parse_u32(&b[0..4]) as i32;
    let month = parse_u32(&b[5..7]) as u32;
    let day = parse_u32(&b[8..10]) as u32;
    let hour = parse_u32(&b[11..13]) as u8;
    let minute = parse_u32(&b[14..16]) as u8;
    let weekday = zeller_weekday(year, month, day);
    let total_minutes = days_from_epoch(year, month, day) * 1440 + (hour as i64) * 60 + minute as i64;
    (hour, weekday, total_minutes)
}

fn parse_u32(b: &[u8]) -> u32 {
    let mut n = 0u32;
    for &c in b {
        n = n * 10 + (c - b'0') as u32;
    }
    n
}

fn zeller_weekday(year: i32, month: u32, day: u32) -> u8 {
    let (y, m) = if month < 3 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let k = y % 100;
    let j = y / 100;
    let h = (day as i32 + (13 * (m as i32 + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
    ((h + 5) % 7) as u8
}

fn days_from_epoch(year: i32, month: u32, day: u32) -> i64 {
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let doy = (153 * (m as i64 - 3) + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mcc() -> MccTable {
        let mut map = HashMap::new();
        map.insert("5411".to_string(), 0.15);
        map.insert("5912".to_string(), 0.20);
        map.insert("7802".to_string(), 0.75);
        MccTable { map }
    }

    #[test]
    fn vectorize_legit_example() {
        let json = r#"{
            "id": "tx-1329056812",
            "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
            "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
            "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
            "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.23},
            "last_transaction": null
        }"#;
        let p: Payload = serde_json::from_str(json).unwrap();
        let mcc = make_mcc();
        let v = vectorize_f32(&p, &mcc);

        assert!((v[0] - 0.004112).abs() < 1e-4);
        assert!((v[1] - 2.0 / 12.0).abs() < 1e-4);
        assert!((v[2] - 0.05).abs() < 1e-4);
        assert!((v[3] - 18.0 / 23.0).abs() < 1e-4);
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
        assert!((v[7] - 0.02923).abs() < 1e-4);
        assert!((v[8] - 0.15).abs() < 1e-4);
        assert_eq!(v[9], 0.0);
        assert_eq!(v[10], 1.0);
        assert_eq!(v[11], 0.0);
        assert!((v[12] - 0.15).abs() < 1e-4);
    }

    #[test]
    fn vectorize_fraud_example() {
        let json = r#"{
            "id": "tx-3330991687",
            "transaction": {"amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z"},
            "customer": {"avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008"]},
            "merchant": {"id": "MERC-068", "mcc": "7802", "avg_amount": 54.86},
            "terminal": {"is_online": false, "card_present": true, "km_from_home": 952.27},
            "last_transaction": null
        }"#;
        let p: Payload = serde_json::from_str(json).unwrap();
        let mcc = make_mcc();
        let v = vectorize_f32(&p, &mcc);

        assert!((v[0] - 0.950597).abs() < 1e-4);
        assert!((v[1] - 10.0 / 12.0).abs() < 1e-4);
        assert!((v[2] - 1.0).abs() < 1e-4);
        assert_eq!(v[8], 1.0);
        assert_eq!(v[11], 1.0);
        assert!((v[12] - 0.75).abs() < 1e-4);
    }

    #[test]
    fn weekday_2026_03_11_is_wednesday() {
        let (_, w, _) = parse_iso("2026-03-11T18:45:53Z");
        assert_eq!(w, 2);
    }

    #[test]
    fn minutes_diff_consistent() {
        let (_, _, a) = parse_iso("2026-03-11T20:23:35Z");
        let (_, _, b) = parse_iso("2026-03-11T14:58:35Z");
        assert_eq!((a - b).unsigned_abs(), 325);
    }
}
