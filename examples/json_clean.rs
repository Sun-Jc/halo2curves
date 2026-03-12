/// Minimal: load JSON scalars, then test GPU in clean state
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_path = args.get(1).expect("usage: <json_path> [limit]");
    let limit: Option<usize> = args.get(2).map(|s| s.parse().unwrap());

    // Load and parse scalars
    eprintln!("loading JSON...");
    let t0 = Instant::now();
    let file = std::fs::File::open(json_path).unwrap();
    let reader = std::io::BufReader::with_capacity(64 * 1024 * 1024, file);
    let val: serde_json::Value = serde_json::from_reader(reader).unwrap();
    let arr = val.get("ra1_dense_hex").unwrap().as_array().unwrap();
    let n = limit.map_or(arr.len(), |l| l.min(arr.len()));
    eprintln!("  parsed in {:.1}s, n={n}", t0.elapsed().as_secs_f64());

    let scalars: Vec<Fr> = arr.iter().take(n)
        .map(|v| parse_fr_from_be_hex(v.as_str().unwrap()))
        .collect();

    // Drop the huge JSON value to free memory
    drop(val);
    eprintln!("  JSON dropped, scalars in memory\n");

    // Generate bases (OsRng)
    eprintln!("generating {n} bases...");
    let bases: Vec<G1Affine> = (0..n).map(|_| G1::random(OsRng).to_affine()).collect();
    eprintln!("  done\n");

    // Test 1: GPU vs CPU with JSON scalars
    eprintln!("--- JSON scalars ---");
    let cpu = msm_best(&scalars, &bases);
    let gpu = msm_gpu::<G1Affine>(&scalars, &bases);
    let ok1 = cpu == gpu;
    eprintln!("  {}", if ok1 { "✅ MATCH" } else { "❌ MISMATCH" });

    // Test 2: same bases, random scalars
    eprintln!("--- Random scalars, same bases ---");
    let rand_scalars: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    let cpu2 = msm_best(&rand_scalars, &bases);
    let gpu2 = msm_gpu::<G1Affine>(&rand_scalars, &bases);
    let ok2 = cpu2 == gpu2;
    eprintln!("  {}", if ok2 { "✅ MATCH" } else { "❌ MISMATCH" });

    // Test 3: JSON scalars again (to see if first call polluted state)
    eprintln!("--- JSON scalars again ---");
    let cpu3 = msm_best(&scalars, &bases);
    let gpu3 = msm_gpu::<G1Affine>(&scalars, &bases);
    let ok3 = cpu3 == gpu3;
    eprintln!("  {}", if ok3 { "✅ MATCH" } else { "❌ MISMATCH" });

    // Was previous GPU run ok?
    eprintln!("\n  gpu1==gpu3 (deterministic): {}", gpu == gpu3);
}
