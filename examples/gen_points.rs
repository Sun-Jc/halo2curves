use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Write},
    iter,
    path::Path,
    time::Duration,
};

use csv::Writer;
use ff::PrimeFieldBits;
use group::{Curve, Group};
use halo2curves::ff::Field;
use halo2curves::{bn256, hash_to_curve::Suite, serde::SerdeObject, CurveAffine, CurveExt};
use rand::RngCore;
use rand_core::OsRng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

fn write_points_to_file<C: CurveExt + SerdeObject>(
    n: usize,
    label: &str,
    path_points: impl AsRef<Path>,
    path_scalars: impl AsRef<Path>,
    path_small_scalars: impl AsRef<Path>,
) where
    C::Scalar: SerdeObject + PrimeFieldBits,
{
    let mut writer_pointer = BufWriter::new(File::create(path_points).unwrap());
    let mut writer_scalars = BufWriter::new(File::create(path_scalars).unwrap());
    let mut writer_small_scalars = BufWriter::new(File::create(path_small_scalars).unwrap());

    let hasher = C::hash_to_curve(label);
    let mut rng = OsRng;

    for i in 0..n {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);

        let point = hasher(&bytes);

        point.write_raw(&mut writer_pointer).unwrap();

        let scalar = C::Scalar::random(&mut rng);
        scalar.write_raw(&mut writer_scalars).unwrap();

        let small_scalar = gen_small_scalar::<C::Scalar>();
        small_scalar.write_raw(&mut writer_small_scalars).unwrap();

        if i % 100000 == 0 {
            println!("Writing {}/{} : {:?} \n{:?}\n{:?}", i, n, point.to_affine(), scalar, small_scalar);
        }
    }
}

fn gen_small_scalar<Scalar: PrimeFieldBits>() -> Scalar {
    let mut rng = OsRng;

    let number: u64 = rng.next_u64();
    let number = number & ((1 << NUM_BITS_SMALL) - 1);
    Scalar::from_u128(number as u128)
}

fn read_points_from_file<C: CurveExt + SerdeObject>(
    path_points: impl AsRef<Path>,
    path_scalars: impl AsRef<Path>,
    path_small_scalars: impl AsRef<Path>,
    start_index: usize,
    n: usize,
) -> (Vec<C>, Vec<C::Scalar>, Vec<C::Scalar>)
where
    C::Scalar: SerdeObject + PrimeFieldBits,
{
    let mut reader_points = BufReader::new(File::open(path_points).unwrap());
    let mut reader_scalars = BufReader::new(File::open(path_scalars).unwrap());
    let mut reader_small_scalars = BufReader::new(File::open(path_small_scalars).unwrap());

    let mut points = Vec::with_capacity(n);
    let mut scalars = Vec::with_capacity(n);
    let mut small_scalars = Vec::with_capacity(n);

    for _ in 0..start_index {
        C::read_raw(&mut reader_points).unwrap();
        C::Scalar::read_raw(&mut reader_scalars).unwrap();
    }

    for _ in 0..n {
        let point = C::read_raw(&mut reader_points).unwrap();
        points.push(point);

        let scalar = C::Scalar::read_raw(&mut reader_scalars).unwrap();
        scalars.push(scalar);

        let small_scalar = C::Scalar::read_raw(&mut reader_small_scalars).unwrap();
        small_scalars.push(small_scalar);
    }

    let small_scalars = small_scalars.par_iter().map(|s| {
        // take only last 20 bits
        let bits = s.to_le_bits();
        let bits = bits.split_at(NUM_BITS_SMALL).0;
        let mut number = 0u64;
        for (i, b) in bits.iter().enumerate() {
            number |= (*b as u64) << i;
        }

        C::Scalar::from(number as u64)
    }).collect();

    (points, scalars, small_scalars)
}

fn write_header<W: Write>(wtr: &mut Writer<W>) -> Result<(), std::io::Error> {
    wtr.write_record(&[
        "ID",
        "BatchSize",
        "NumSmallScalar",
        "MSMTime(ns)",
        "RngId",
        "ApproachId",
        "EnvType",
        "SizeBitsSmallScalar",
        "Res",
    ])?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Record {
    id: usize,
    batch_size: usize,
    num_small_scalar: usize,
    msm_time_ns: u128,
    rng_id: usize,
    approach_id: usize,
    env_type: usize, // 0 for amd64, 1 for arm64
    size_bits_small_scalar: usize,
    res: Vec<u8>,
}

fn write_one_record<W: Write>(wtr: &mut Writer<W>, record: Record) -> Result<(), std::io::Error> {
    let Record {
        id,
        batch_size,
        num_small_scalar,
        msm_time_ns: msm_time,
        rng_id,
        approach_id,
        env_type,
        size_bits_small_scalar,
        res,
    } = record;

    wtr.write_record(&[
        id.to_string(),
        batch_size.to_string(),
        num_small_scalar.to_string(),
        msm_time.to_string(),
        rng_id.to_string(),
        approach_id.to_string(),
        env_type.to_string(),
        size_bits_small_scalar.to_string(),
        hex::encode(res),
    ])?;
    wtr.flush()?;
    Ok(())
}

const NUM_BITS_SMALL: usize = 16;

fn is_small<Scalar: PrimeFieldBits>(s: &Scalar) -> bool {
    let bits = s.to_le_bits();
    bits.split_at(NUM_BITS_SMALL)
        .1
        .iter()
        .all(|b| *b as u8 == 0)
}

#[inline]
fn benchmark<C: Curve + SerdeObject>(
    msm_res: C,
    scalars: &[C::Scalar],
    dur: Duration,
    id: usize,
    rng_id: usize,
    approach_id: usize,
    env_type: usize,
) -> Record
where
    C::Scalar: PrimeFieldBits,
{
    let num_small_scalar = scalars.par_iter().filter(|s| is_small(*s)).count();

    let record = Record {
        id,
        batch_size: scalars.len(),
        num_small_scalar,
        msm_time_ns: dur.as_nanos(),
        rng_id,
        approach_id,
        env_type,
        size_bits_small_scalar: NUM_BITS_SMALL,
        res: msm_res.to_raw_bytes(),
    };

    record
}

fn benchmark_all() {
    // type Scalar = <halo2curves::bn256::G1 as Group>::Scalar;
    type Point = bn256::G1;

    const NUM_REPEAT: usize = 5;

    let mut cnt = 0;
    let batch_sizes: [usize; 6] = [1 << 10, 1<<15, 1 << 20, 1<<21, 1<<22, 1<<23];
    let rng_ids = [1, 2, 3];

    let approach_id = 0;
    let env_type = 0;

    let mut writer = Writer::from_path("benchmark.csv").unwrap();
    write_header(&mut writer).unwrap();

    for rng_id in rng_ids {
        let (points, scalars, small_scalars) = read_points_from_file::<Point>(
            format!("/home/ubuntu/points.{}.bin", rng_id),
            format!("/home/ubuntu/scalars.{}.bin", rng_id),
            format!("/home/ubuntu/small_scalars.{}.bin", rng_id),
            0,
            *batch_sizes.last().unwrap(),
        );

        let points = points.par_iter().map(|p| p.to_affine()).collect::<Vec<_>>();

        for batch_size in batch_sizes {
            let points = &points[..batch_size];
            for num_small in [0, (batch_size as f64 * 0.95) as usize, batch_size] {
                let mut final_scalars = small_scalars.iter().take(num_small).cloned().collect::<Vec<_>>();
                final_scalars.extend(scalars.iter().cloned().take(batch_size - num_small));

                let scalars = &final_scalars;

                assert_eq!(points.len(), scalars.len());
                assert_eq!(points.len(), batch_size);

                for _ in 0..NUM_REPEAT {
                    let id = cnt;
                    cnt += 1;

                    let start = std::time::Instant::now();

                    let res = if num_small == batch_size {
                        halo2curves::msm::msm_best2(scalars, points)
                    } else {
                        halo2curves::msm::msm_best(scalars, points)
                    };

                    let dur = start.elapsed();

                    let record = benchmark(res, scalars, dur, id, rng_id, 1, env_type);

                    write_one_record(&mut writer, record.clone()).unwrap();


                    {
                        if batch_size == num_small {
                            let now = std::time::Instant::now();
                            let res2 = halo2curves::msm::msm_best(scalars, points);
                            let dur2 = now.elapsed();
                            if res != res2 {
                                panic!("Results do not match!");
                            } else {
                                println!("Results match! {:?} / {:?}, {}", dur, dur2, dur2.as_nanos() / dur.as_nanos());
                            }

                            let id = cnt;
                            cnt += 1 ;

                            let record = benchmark(res2, scalars, dur2, id, rng_id, approach_id, env_type);
                            write_one_record(&mut writer, record.clone()).unwrap();
                        }
                    }


                    println!(
                        "ID: {}, BatchSize: {}, NumSmallScalar: {}, MSMTime: {} ns, RngId: {}, EnvType: {}",
                        id,
                        batch_size,
                        record.num_small_scalar,
                        record.msm_time_ns,
                        rng_id,
                        env_type
                    );
                }
            }
        }
    }
}

fn main() {
    let n = 1 << 35;

    let args: Vec<String> = env::args().collect();
    let id = &args[1];

    let path_points = &format!("points.{}.bin", id);
    let path_scalars = &format!("scalars.{}.bin", id);
    let path_small_scalars = &format!("small_scalars.{}.bin", id);

    write_points_to_file::<bn256::G1>(
        n,
        &format!("test{}", id),
        path_points,
        path_scalars,
        path_small_scalars,
    );

    // let read_points = read_points_from_file::<bn256::G1>(path_points, path_scalars, 0, 1 << 20);
    // println!("Successfully generated and read {} points.", n);
}
