/// Isolate: which scalar values cause GPU mismatch at n=262144?
use ff::{Field, PrimeField};
use group::{Curve, Group};
use halo2curves::bn256::{Fr, G1, G1Affine};
use halo2curves::gpu::msm_gpu;
use halo2curves::msm::msm_best;
use rand_core::OsRng;

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(10 + c - b'a'),
        b'A'..=b'F' => Some(10 + c - b'A'),
        _ => None,
    }
}

fn parse_fr(s: &str) -> Fr {
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

fn test(scalars: &[Fr], bases: &[G1Affine]) -> bool {
    msm_best(scalars, bases) == msm_gpu::<G1Affine>(scalars, bases)
}

fn main() {
    let json_path = std::env::args().nth(1).expect("usage: <json_path>");

    eprintln!("loading JSON...");
    let file = std::fs::File::open(&json_path).unwrap();
    let reader = std::io::BufReader::with_capacity(64 * 1024 * 1024, file);
    let val: serde_json::Value = serde_json::from_reader(reader).unwrap();
    let arr = val.get("ra1_dense_hex").unwrap().as_array().unwrap();

    let n = 262144; // exact failure boundary
    let json_scalars: Vec<Fr> = arr.iter().take(n)
        .map(|v| parse_fr(v.as_str().unwrap()))
        .collect();
    drop(val);

    let bases: Vec<G1Affine> = (0..n).map(|_| G1::random(OsRng).to_affine()).collect();

    eprintln!("\nn={n}, c should be 15\n");

    // 1. Original JSON scalars
    eprintln!("1. Original JSON:       {}", if test(&json_scalars, &bases) { "✅" } else { "❌" });

    // 2. All random
    let rand_s: Vec<Fr> = (0..n).map(|_| Fr::random(OsRng)).collect();
    eprintln!("2. All random:          {}", if test(&rand_s, &bases) { "✅" } else { "❌" });

    // 3. Keep JSON zero positions, replace non-zero with random
    let mut nz_rand = json_scalars.clone();
    for s in &mut nz_rand {
        if !bool::from(s.is_zero()) { *s = Fr::random(OsRng); }
    }
    eprintln!("3. Zero pattern + rand: {}", if test(&nz_rand, &bases) { "✅" } else { "❌" });

    // 4. Keep JSON non-zero values exactly, but pad all zeros to n (remove some zeros)
    // Actually: just keep the exact scalars as-is
    // 5. Replace other (non-0, non-1) with Fr::from(2u64) (a small scalar)
    let zero = Fr::ZERO;
    let one = Fr::ONE;
    let two = Fr::from(2u64);
    let mut small_other = json_scalars.clone();
    for s in &mut small_other {
        if *s != zero && *s != one { *s = two; }
    }
    eprintln!("5. Other→2:             {}", if test(&small_other, &bases) { "✅" } else { "❌" });

    // 6. Use just ONE of the "other" values repeated
    let other_val = parse_fr("0x1177972103ffe9a2b3ed5ae0a1b4f152c47528dad33c60fb66a1717293d5eb52");
    let mut single_other = json_scalars.clone();
    for s in &mut single_other {
        if *s != zero && *s != one { *s = other_val; }
    }
    eprintln!("6. Other→single big:    {}", if test(&single_other, &bases) { "✅" } else { "❌" });

    // 7. Use the "negative" value
    let neg_val = parse_fr("0x2dcc2bf57a66f060e8659ea0e67c1999cbf5ebe2e1de2e44ae25e48d548f5e71");
    let mut single_neg = json_scalars.clone();
    for s in &mut single_neg {
        if *s != zero && *s != one { *s = neg_val; }
    }
    eprintln!("7. Other→single neg:    {}", if test(&single_neg, &bases) { "✅" } else { "❌" });

    // 8. What if we use many distinct "other" values but all small (< 2^128)?
    let mut small_rand_other = json_scalars.clone();
    let mut ctr = 0u64;
    for s in &mut small_rand_other {
        if *s != zero && *s != one {
            ctr += 1;
            *s = Fr::from(ctr + 100);
        }
    }
    eprintln!("8. Other→small seq:     {}", if test(&small_rand_other, &bases) { "✅" } else { "❌" });

    // 9. Test at n=262143 (one less, should pass)
    eprintln!("\n9. n=262143:            {}", if test(&json_scalars[..262143], &bases[..262143]) { "✅" } else { "❌" });
    eprintln!("10.n=262144:            {}", if test(&json_scalars, &bases) { "✅" } else { "❌" });

    // 11. Does the last scalar matter?
    let mut without_last = json_scalars.clone();
    without_last[262143] = Fr::ZERO;
    eprintln!("11.last→0:              {}", if test(&without_last, &bases) { "✅" } else { "❌" });
}
