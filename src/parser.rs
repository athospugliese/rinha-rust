use crate::ivf::DIM;
use crate::vectorizer::MccTable;

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn parse_and_vectorize(body: &[u8], mcc: &MccTable) -> Option<[i16; DIM]> {
    let tx = locate_object(body, b"\"transaction\"")?;
    let cust = locate_object(body, b"\"customer\"")?;
    let merch = locate_object(body, b"\"merchant\"")?;
    let term = locate_object(body, b"\"terminal\"")?;
    let last = locate_object_or_null(body, b"\"last_transaction\"");

    let amount = read_number(&body[tx.0..tx.1], b"\"amount\"")?;
    let installments = read_number(&body[tx.0..tx.1], b"\"installments\"")?;
    let requested_at = read_string(&body[tx.0..tx.1], b"\"requested_at\"")?;

    let avg_amount = read_number(&body[cust.0..cust.1], b"\"avg_amount\"")?;
    let tx_count_24h = read_number(&body[cust.0..cust.1], b"\"tx_count_24h\"")?;
    let merchant_id = read_string(&body[merch.0..merch.1], b"\"id\"")?;
    let in_known = array_has(&body[cust.0..cust.1], b"\"known_merchants\"", merchant_id);

    let mcc_code = read_string(&body[merch.0..merch.1], b"\"mcc\"")?;
    let merchant_avg = read_number(&body[merch.0..merch.1], b"\"avg_amount\"")?;

    let is_online = read_bool(&body[term.0..term.1], b"\"is_online\"")?;
    let card_present = read_bool(&body[term.0..term.1], b"\"card_present\"")?;
    let km_home = read_number(&body[term.0..term.1], b"\"km_from_home\"")?;

    let mut v = [0f32; DIM];
    v[0] = clamp01(amount / 10_000.0);
    v[1] = clamp01(installments / 12.0);
    v[2] = clamp01((amount / avg_amount) / 10.0);

    let (hour_a, weekday_a, ts_a) = parse_iso(requested_at);
    v[3] = hour_a as f32 / 23.0;
    v[4] = weekday_a as f32 / 6.0;

    if let Some((s, e)) = last {
        let ts_str = read_string(&body[s..e], b"\"timestamp\"")?;
        let km_last = read_number(&body[s..e], b"\"km_from_current\"")?;
        let (_, _, ts_b) = parse_iso(ts_str);
        let delta = (ts_a as i64 - ts_b as i64).unsigned_abs() as f32;
        v[5] = clamp01(delta / 1440.0);
        v[6] = clamp01(km_last / 1000.0);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    v[7] = clamp01(km_home / 1000.0);
    v[8] = clamp01(tx_count_24h / 20.0);
    v[9] = if is_online { 1.0 } else { 0.0 };
    v[10] = if card_present { 1.0 } else { 0.0 };
    v[11] = if in_known { 0.0 } else { 1.0 };
    v[12] = mcc.lookup_bytes(mcc_code);
    v[13] = clamp01(merchant_avg / 10_000.0);

    let mut out = [0i16; DIM];
    for i in 0..DIM {
        let s = v[i].clamp(-1.0, 1.0) * 10_000.0;
        out[i] = s.round() as i16;
    }
    Some(out)
}

fn locate_object<'a>(buf: &'a [u8], key: &[u8]) -> Option<(usize, usize)> {
    let pos = find_subseq(buf, key)?;
    let mut i = pos + key.len();
    while i < buf.len() && (buf[i] == b' ' || buf[i] == b'\t' || buf[i] == b':' || buf[i] == b'\r' || buf[i] == b'\n') {
        i += 1;
    }
    if i >= buf.len() || buf[i] != b'{' {
        return None;
    }
    let start = i;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    while i < buf.len() {
        let c = buf[i];
        if in_str {
            if esc { esc = false; }
            else if c == b'\\' { esc = true; }
            else if c == b'"' { in_str = false; }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((start, i + 1));
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn locate_object_or_null(buf: &[u8], key: &[u8]) -> Option<(usize, usize)> {
    let pos = find_subseq(buf, key)?;
    let mut i = pos + key.len();
    while i < buf.len() && (buf[i] == b' ' || buf[i] == b'\t' || buf[i] == b':' || buf[i] == b'\r' || buf[i] == b'\n') {
        i += 1;
    }
    if i >= buf.len() {
        return None;
    }
    if buf[i] == b'n' {
        return None;
    }
    locate_object(&buf[pos..], key).map(|(s, e)| (s + pos, e + pos))
}

fn read_number(buf: &[u8], key: &[u8]) -> Option<f32> {
    let pos = find_subseq(buf, key)?;
    let after_colon = skip_to_value(buf, pos + key.len())?;
    let bytes = number_bytes(buf, after_colon);
    let s = std::str::from_utf8(&buf[after_colon..after_colon + bytes]).ok()?;
    s.parse::<f32>().ok()
}

fn read_string<'a>(buf: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    let pos = find_subseq(buf, key)?;
    let after_colon = skip_to_value(buf, pos + key.len())?;
    if buf[after_colon] != b'"' {
        return None;
    }
    let start = after_colon + 1;
    let mut end = start;
    while end < buf.len() && buf[end] != b'"' {
        end += 1;
    }
    Some(&buf[start..end])
}

fn read_bool(buf: &[u8], key: &[u8]) -> Option<bool> {
    let pos = find_subseq(buf, key)?;
    let after_colon = skip_to_value(buf, pos + key.len())?;
    if buf[after_colon..].starts_with(b"true") {
        Some(true)
    } else if buf[after_colon..].starts_with(b"false") {
        Some(false)
    } else {
        None
    }
}

fn array_has(buf: &[u8], key: &[u8], needle: &[u8]) -> bool {
    let pos = match find_subseq(buf, key) {
        Some(p) => p,
        None => return false,
    };
    let mut i = pos + key.len();
    while i < buf.len() && (buf[i] == b' ' || buf[i] == b'\t' || buf[i] == b':') {
        i += 1;
    }
    if i >= buf.len() || buf[i] != b'[' {
        return false;
    }
    i += 1;
    while i < buf.len() && buf[i] != b']' {
        if buf[i] == b'"' {
            let s = i + 1;
            let mut e = s;
            while e < buf.len() && buf[e] != b'"' {
                e += 1;
            }
            if &buf[s..e] == needle {
                return true;
            }
            i = e + 1;
        } else {
            i += 1;
        }
    }
    false
}

fn skip_to_value(buf: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i < buf.len() && buf[i] != b':' {
        i += 1;
    }
    if i >= buf.len() {
        return None;
    }
    i += 1;
    while i < buf.len() && (buf[i] == b' ' || buf[i] == b'\t' || buf[i] == b'\r' || buf[i] == b'\n') {
        i += 1;
    }
    if i >= buf.len() {
        None
    } else {
        Some(i)
    }
}

fn number_bytes(buf: &[u8], from: usize) -> usize {
    let mut i = from;
    while i < buf.len() {
        let c = buf[i];
        if c.is_ascii_digit() || c == b'.' || c == b'-' || c == b'+' || c == b'e' || c == b'E' {
            i += 1;
        } else {
            break;
        }
    }
    i - from
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    memchr::memmem::find(haystack, needle)
}

fn parse_iso(b: &[u8]) -> (u8, u8, i64) {
    let year = parse_u32(&b[0..4]) as i32;
    let month = parse_u32(&b[5..7]);
    let day = parse_u32(&b[8..10]);
    let hour = parse_u32(&b[11..13]) as u8;
    let minute = parse_u32(&b[14..16]) as u8;
    let weekday = zeller_weekday(year, month, day);
    let total = days_from_epoch(year, month, day) * 1440 + (hour as i64) * 60 + minute as i64;
    (hour, weekday, total)
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
