/// Binary search for mismatch: which portion of JSON scalars triggers the GPU bug?
use ff::{Field, PrimeField};
use group::{Curve, Group};
use halo2curves::bn256::{Fr, G1, G1Affine};
use halo2curves::gpu::msm_gpu;
use halo2curves::msm::msm_best;
use rand_core::OsRng;
use std::time::Instant;

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(10 + c - b'a'),
        b'A'..=b'F' => Some(10 + c - b'A'),
        _ => None,
    }
}

fn parse_fr_from_be_hex(s: &str) -> Fr {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let mut bytes = s.as_bytes().to_vec();
    if bytes.len() % 2 == 1 { bytes.insert(0, b'0'); }
    let mut be = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_nibble(bytes[i]).unwrap();
        let lo = hex_nibble(bytes[i + 1]).unwrap();
        be.push((hi << 4) | lo);
    }
    let mut repr = <Fr as PrimeField>::Repr::default();
    let le = repr.as_mut();
    for (i, b) in be.iter().rev().enumerate() { le[i] = *b; }
    Fr::from_repr(repr).unwrap()
}

fn test_gpu_cpu(scalars: &[Fr], bases: &[G1Affine]) -> bool {
    let cpu = msm_best(scalars, bases);
    let gpu = msm_gpu::<G1Affine>(scalars, bases);
    cpu == gpu
}

fn main() {
    let json_path = std::env::args().nth(1).expect("usage: <json_path>");

    eprintln!("loading JSON...");
    let t0 = Instant::now();
    let file = std::fs::File::open(&json_path).unwrap();
    let reader = std::io::BufReader::with_capacity(64 * 1024 * 1024, file);
    let val: serde_json::Value = serde_json::from_reader(reader).unwrap();
    let arr = val.get("ra1_dense_hex").unwrap().as_array().unwrap();
    let n = 300000.min(arr.len());
    eprintln!("  parsed in {:.1}s", t0.elapsed().as_secs_f64());

    let json_scalars: Vec<Fr> = arr.iter().take(n)
        .map(|v| parse_fr_from_be_hex(v.as_str().unwrap()))
        .collect();
    drop(val);

    eprintln!("generating {n} bases...");
    let bases: Vec<G1Affine> = (0..n).map(|_| G1::random(OsRng).to_affine()).collect();

    // Verify JSON scalars fail
    eprintln!("\n--- Full JSON scalars (n={n}) ---");
    let ok = test_gpu_cpu(&json_scalars, &bases);
    eprintln!("  {}", if ok { "✅" } else { "❌" });
    if ok { eprintln!("  (passed! bug doesn't reproduce)"); return; }

    // Check if it's reproducible
    let ok2 = test_gpu_cpu(&json_scalars, &bases);
    eprintln!("  second run: {}", if ok2 { "✅" } else { "❌" });

    // What fraction of the scalars are Fr::ZERO vs Fr::ONE vs other?
    let zero = Fr::ZERO;
    let one = Fr::ONE;
    let mut cnt_zero = 0usize;
    let mut cnt_one = 0usize;
    let mut cnt_other = 0usize;
    for s in &json_scalars {
        if *s == zero { cnt_zero += 1; }
        else if *s == one { cnt_one += 1; }
        else { cnt_other += 1; }
    }
    eprintln!("\n  scalar distribution: zero={cnt_zero}, one={cnt_one}, other={cnt_other}");

    // Try replacing all zeros with Fr::ZERO (same) and all ones with Fr::ONE (same)
    // but randomize the "other" values
    eprintln!("\n--- Replace 'other' scalars with random ---");
    let mut modified = json_scalars.clone();
    for s in &mut modified {
        if *s != zero && *s != one {
            *s = Fr::random(OsRng);
        }
    }
    let ok = test_gpu_cpu(&modified, &bases);
    eprintln!("  {}", if ok { "✅" } else { "❌" });

    // Try: make all scalars = JSON pattern but use only {0, 1}
    eprintln!("--- Replace all with 0/1 based on zero/nonzero ---");
    let binary: Vec<Fr> = json_scalars.iter().map(|s| {
        if *s == zero { Fr::ZERO } else { Fr::ONE }
    }).collect();
    let ok = test_gpu_cpu(&binary, &bases);
    eprintln!("  {}", if ok { "✅" } else { "❌" });

    // Binary search by size: at what n does it start failing?
    eprintln!("\n--- Binary search on n ---");
    let mut lo = 16384;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let ok = test_gpu_cpu(&json_scalars[..mid], &bases[..mid]);
        eprintln!("  n={mid}: {}", if ok { "✅" } else { "❌" });
        if ok {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    eprintln!("  first failure at n={lo}");
}
