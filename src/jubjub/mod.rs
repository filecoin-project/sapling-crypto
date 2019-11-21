//! Jubjub is a twisted Edwards curve defined over the BLS12-381 scalar
//! field, Fr. It takes the form `-x^2 + y^2 = 1 + dx^2y^2` with
//! `d = -(10240/10241)`. It is birationally equivalent to a Montgomery
//! curve of the form `y^2 = x^3 + Ax^2 + x` with `A = 40962`. This
//! value `A` is the smallest integer choice such that:
//!
//! * `(A - 2) / 4` is a small integer (`10240`).
//! * `A^2 - 4` is quadratic nonresidue.
//! * The group order of the curve and its quadratic twist has a large
//!   prime factor.
//!
//! Jubjub has `s = 0x0e7db4ea6533afa906673b0101343b00a6682093ccc81082d0970e5ed6f72cb7`
//! as the prime subgroup order, with cofactor 8. (The twist has
//! cofactor 4.)
//!
//! It is a complete twisted Edwards curve, so the equivalence with
//! the Montgomery curve forms a group isomorphism, allowing points
//! to be freely converted between the two forms.
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::Path;

use ff::{Field, PrimeField, SqrtField};
use lazy_static::lazy_static;
use paired::bls12_381::{Bls12, Fr};
use paired::Engine;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::constants;
use crate::group_hash::group_hash;

/// This is an implementation of the twisted Edwards Jubjub curve.
pub mod edwards;

/// This is an implementation of the birationally equivalent
/// Montgomery curve.
pub mod montgomery;

/// This is an implementation of the scalar field for Jubjub.
pub mod fs;

#[cfg(test)]
pub mod tests;

pub const MAX_EXP_TABLE_SEGMENTS_IN_MEMORY: usize = 50;

lazy_static! {
    // The interface for the Edward's curve group law requires that the curve's `d` parameter be
    // passed in using a type that implements `JubjubParams`. This static value creates a structure
    // that contains only the Edward's curve `d` parameter, which can then be passed into Edward's
    // point addition (and multiplication).
    static ref JUBJUB_BLS12_PARAMS_WITH_EDWARDS_D: JubjubBls12 = JubjubBls12 {
        edwards_d: Fr::from_str("19257038036680949359750312669786877991949435402254120286184196891950884077233").unwrap(),
        pedersen_hash_exp_window_size_supplied: 0,
        pedersen_hash_exp: vec![],
        montgomery_a: Fr::zero(),
        montgomery_2a: Fr::zero(),
        scale: Fr::zero(),
        pedersen_hash_generators: vec![],
        pedersen_circuit_generators: vec![],
        fixed_base_generators: vec![],
        fixed_base_circuit_generators: vec![],
        exp_table_path: None,
    };
}

/// Point of unknown order.
pub enum Unknown {}

/// Point of prime order.
#[derive(Clone)]
pub enum PrimeOrder {}

/// Fixed generators of the Jubjub curve of unknown
/// exponent.
#[derive(Copy, Clone)]
pub enum FixedGenerators {
    /// The prover will demonstrate knowledge of discrete log
    /// with respect to this base when they are constructing
    /// a proof, in order to authorize proof construction.
    ProofGenerationKey = 0,

    /// The note commitment is randomized over this generator.
    NoteCommitmentRandomness = 1,

    /// The node commitment is randomized again by the position
    /// in order to supply the nullifier computation with a
    /// unique input w.r.t. the note being spent, to prevent
    /// Faerie gold attacks.
    NullifierPosition = 2,

    /// The value commitment is used to check balance between
    /// inputs and outputs. The value is placed over this
    /// generator.
    ValueCommitmentValue = 3,
    /// The value commitment is randomized over this generator,
    /// for privacy.
    ValueCommitmentRandomness = 4,

    /// The spender proves discrete log with respect to this
    /// base at spend time.
    SpendingKeyGenerator = 5,

    Max = 6,
}

pub trait ToUniform {
    fn to_uniform(digest: &[u8]) -> Self;
}

/// This is an extension to the pairing Engine trait which
/// offers a scalar field for the embedded curve (Jubjub)
/// and some pre-computed parameters.
pub trait JubjubEngine: Engine {
    /// The scalar field of the Jubjub curve
    type Fs: PrimeField + SqrtField + ToUniform;
    /// The parameters of Jubjub and the Sapling protocol
    type Params: JubjubParams<Self>;
}

/// The pre-computed parameters for Jubjub, including curve
/// constants and various limits and window tables.
pub trait JubjubParams<E: JubjubEngine>: Sized {
    /// The `d` constant of the twisted Edwards curve.
    fn edwards_d(&self) -> &E::Fr;
    /// The `A` constant of the birationally equivalent Montgomery curve.
    fn montgomery_a(&self) -> &E::Fr;
    /// The `A` constant, doubled.
    fn montgomery_2a(&self) -> &E::Fr;
    /// The scaling factor used for conversion from the Montgomery form.
    fn scale(&self) -> &E::Fr;
    /// Returns the generators (for each segment) used in all Pedersen commitments.
    fn pedersen_hash_generators(&self) -> &[edwards::Point<E, PrimeOrder>];
    /// Returns the exp table for Pedersen hashes.
    fn pedersen_hash_exp_table(&self) -> &[Vec<Vec<edwards::Point<E, PrimeOrder>>>];
    /// Returns the maximum number of chunks per segment of the Pedersen hash.
    fn pedersen_hash_chunks_per_generator(&self) -> usize;
    /// Returns the pre-computed window tables [-4, 3, 2, 1, 1, 2, 3, 4] of different
    /// magnitudes of the Pedersen hash segment generators.
    fn pedersen_circuit_generators(&self) -> &[Vec<Vec<(E::Fr, E::Fr)>>];

    /// Returns the number of chunks needed to represent a full scalar during fixed-base
    /// exponentiation.
    fn fixed_base_chunks_per_generator(&self) -> usize;
    /// Returns a fixed generator.
    fn generator(&self, base: FixedGenerators) -> &edwards::Point<E, PrimeOrder>;
    /// Returns a window table [0, 1, ..., 8] for different magnitudes of some
    /// fixed generator.
    fn circuit_generators(&self, _: FixedGenerators) -> &[Vec<(E::Fr, E::Fr)>];
    /// Returns the window size for exponentiation of Pedersen hash generators
    /// outside the circuit
    fn pedersen_hash_exp_window_size(&self) -> u32;

    /// If these parameters use a file-backed Pedersen Hash exp-table, returns the path to the
    /// exp-table file.
    fn exp_table_path(&self) -> &Option<String>;
}

impl JubjubEngine for Bls12 {
    type Fs = self::fs::Fs;
    type Params = JubjubBls12;
}

#[derive(Clone)]
pub struct JubjubBls12 {
    edwards_d: Fr,
    montgomery_a: Fr,
    montgomery_2a: Fr,
    scale: Fr,

    pedersen_hash_generators: Vec<edwards::Point<Bls12, PrimeOrder>>,
    pedersen_hash_exp: Vec<Vec<Vec<edwards::Point<Bls12, PrimeOrder>>>>,
    pedersen_circuit_generators: Vec<Vec<Vec<(Fr, Fr)>>>,

    fixed_base_generators: Vec<edwards::Point<Bls12, PrimeOrder>>,
    fixed_base_circuit_generators: Vec<Vec<Vec<(Fr, Fr)>>>,
    pedersen_hash_exp_window_size_supplied: u32,

    exp_table_path: Option<String>,
}

impl JubjubParams<Bls12> for JubjubBls12 {
    fn edwards_d(&self) -> &Fr {
        &self.edwards_d
    }
    fn montgomery_a(&self) -> &Fr {
        &self.montgomery_a
    }
    fn montgomery_2a(&self) -> &Fr {
        &self.montgomery_2a
    }
    fn scale(&self) -> &Fr {
        &self.scale
    }
    fn pedersen_hash_generators(&self) -> &[edwards::Point<Bls12, PrimeOrder>] {
        &self.pedersen_hash_generators
    }
    fn pedersen_hash_exp_table(&self) -> &[Vec<Vec<edwards::Point<Bls12, PrimeOrder>>>] {
        &self.pedersen_hash_exp
    }
    fn pedersen_hash_chunks_per_generator(&self) -> usize {
        63
    }
    fn fixed_base_chunks_per_generator(&self) -> usize {
        84
    }
    fn pedersen_circuit_generators(&self) -> &[Vec<Vec<(Fr, Fr)>>] {
        &self.pedersen_circuit_generators
    }
    fn generator(&self, base: FixedGenerators) -> &edwards::Point<Bls12, PrimeOrder> {
        &self.fixed_base_generators[base as usize]
    }
    fn circuit_generators(&self, base: FixedGenerators) -> &[Vec<(Fr, Fr)>] {
        &self.fixed_base_circuit_generators[base as usize][..]
    }
    fn pedersen_hash_exp_window_size(&self) -> u32 {
        self.pedersen_hash_exp_window_size_supplied
    }

    fn exp_table_path(&self) -> &Option<String> {
        &self.exp_table_path
    }
}

impl JubjubBls12 {
    pub fn new() -> Self {
        Self::new_with_n_segments_and_window_size(5, 8, None).unwrap()
    }

    pub fn new_with_window_size(window_size: u32) -> Self {
        Self::new_with_n_segments_and_window_size(5, window_size, None).unwrap()
    }

    /// Returns an `Err` if we failed to create `exp_table_path` (create the directories in the path
    /// that do not exist) or if we fail to create the exp-table file. An `Err` will never be
    /// returned if the `exp_table_path` argument is `None`.
    pub fn new_with_n_segments_and_window_size(
        n_segments: usize,
        window_size: u32,
        exp_table_path: Option<&str>,
    ) -> io::Result<Self> {
        let montgomery_a = Fr::from_str("40962").unwrap();
        let mut montgomery_2a = montgomery_a;
        montgomery_2a.double();

        let mut tmp_params = JubjubBls12 {
            // d = -(10240/10241)
            edwards_d: Fr::from_str(
                "19257038036680949359750312669786877991949435402254120286184196891950884077233",
            )
            .unwrap(),
            // A = 40962
            montgomery_a: montgomery_a,
            // 2A = 2.A
            montgomery_2a: montgomery_2a,
            // scaling factor = sqrt(4 / (a - d))
            scale: Fr::from_str(
                "17814886934372412843466061268024708274627479829237077604635722030778476050649",
            )
            .unwrap(),

            // We'll initialize these vectors below.
            pedersen_hash_generators: vec![],
            pedersen_hash_exp: vec![],
            pedersen_circuit_generators: vec![],
            fixed_base_generators: vec![],
            fixed_base_circuit_generators: vec![],

            pedersen_hash_exp_window_size_supplied: window_size,
            exp_table_path: exp_table_path.map(|path| path.to_string()),
        };

        fn find_group_hash<E: JubjubEngine>(
            m: &[u8],
            personalization: &[u8; 8],
            params: &E::Params,
        ) -> edwards::Point<E, PrimeOrder> {
            let mut tag = m.to_vec();
            let i = tag.len();
            tag.push(0u8);

            loop {
                let gh = group_hash(&tag, personalization, params);

                // We don't want to overflow and start reusing generators
                assert!(tag[i] != u8::max_value());
                tag[i] += 1;

                if let Some(gh) = gh {
                    break gh;
                }
            }
        }

        // Create the bases for the Pedersen hashes
        {
            let mut pedersen_hash_generators = vec![];

            for m in 0..n_segments as u32 {
                use byteorder::{LittleEndian, WriteBytesExt};

                let mut segment_number = [0u8; 4];
                (&mut segment_number[0..4])
                    .write_u32::<LittleEndian>(m)
                    .unwrap();

                pedersen_hash_generators.push(find_group_hash(
                    &segment_number,
                    constants::PEDERSEN_HASH_GENERATORS_PERSONALIZATION,
                    &tmp_params,
                ));
            }

            // Check for duplicates, far worse than spec inconsistencies!
            for (i, p1) in pedersen_hash_generators.iter().enumerate() {
                if p1 == &edwards::Point::zero() {
                    panic!("Neutral element!");
                }

                for p2 in pedersen_hash_generators.iter().skip(i + 1) {
                    if p1 == p2 {
                        panic!("Duplicate generator!");
                    }
                }
            }

            tmp_params.pedersen_hash_generators = pedersen_hash_generators;
        }

        // Create the exp table for the Pedersen hash generators
        if let Some(path) = exp_table_path {
            if must_write_exp_table(n_segments, window_size, path) {
                write_exp_table_to_file(&tmp_params.pedersen_hash_generators, window_size, path)?;
            }
            // As an optimization, always keep the first 5 segments worth of the exp-table in main
            // memory.
            tmp_params.pedersen_hash_exp = create_exp_table_for_generators(
                &tmp_params.pedersen_hash_generators[..5],
                window_size,
            );
        } else {
            tmp_params.pedersen_hash_exp =
                create_exp_table_for_generators(&tmp_params.pedersen_hash_generators, window_size);
        }

        // Create the bases for other parts of the protocol
        {
            let mut fixed_base_generators =
                vec![edwards::Point::zero(); FixedGenerators::Max as usize];

            fixed_base_generators[FixedGenerators::ProofGenerationKey as usize] = find_group_hash(
                &[],
                constants::PROOF_GENERATION_KEY_BASE_GENERATOR_PERSONALIZATION,
                &tmp_params,
            );

            fixed_base_generators[FixedGenerators::NoteCommitmentRandomness as usize] =
                find_group_hash(
                    b"r",
                    constants::PEDERSEN_HASH_GENERATORS_PERSONALIZATION,
                    &tmp_params,
                );

            fixed_base_generators[FixedGenerators::NullifierPosition as usize] = find_group_hash(
                &[],
                constants::NULLIFIER_POSITION_IN_TREE_GENERATOR_PERSONALIZATION,
                &tmp_params,
            );

            fixed_base_generators[FixedGenerators::ValueCommitmentValue as usize] = find_group_hash(
                b"v",
                constants::VALUE_COMMITMENT_GENERATOR_PERSONALIZATION,
                &tmp_params,
            );

            fixed_base_generators[FixedGenerators::ValueCommitmentRandomness as usize] =
                find_group_hash(
                    b"r",
                    constants::VALUE_COMMITMENT_GENERATOR_PERSONALIZATION,
                    &tmp_params,
                );

            fixed_base_generators[FixedGenerators::SpendingKeyGenerator as usize] = find_group_hash(
                &[],
                constants::SPENDING_KEY_GENERATOR_PERSONALIZATION,
                &tmp_params,
            );

            // Check for duplicates, far worse than spec inconsistencies!
            for (i, p1) in fixed_base_generators.iter().enumerate() {
                if p1 == &edwards::Point::zero() {
                    panic!("Neutral element!");
                }

                for p2 in fixed_base_generators.iter().skip(i + 1) {
                    if p1 == p2 {
                        panic!("Duplicate generator!");
                    }
                }
            }

            tmp_params.fixed_base_generators = fixed_base_generators;
        }

        // Create the 2-bit window table lookups for each 4-bit
        // "chunk" in each segment of the Pedersen hash
        {
            let mut pedersen_circuit_generators = vec![];

            // Process each segment
            for gen in tmp_params.pedersen_hash_generators.iter().cloned() {
                let mut gen = montgomery::Point::from_edwards(&gen, &tmp_params);
                let mut windows = vec![];
                for _ in 0..tmp_params.pedersen_hash_chunks_per_generator() {
                    // Create (x, y) coeffs for this chunk
                    let mut coeffs = vec![];
                    let mut g = gen.clone();

                    // coeffs = g, g*2, g*3, g*4
                    for _ in 0..4 {
                        coeffs.push(g.into_xy().expect("cannot produce O"));
                        g = g.add(&gen, &tmp_params);
                    }
                    windows.push(coeffs);

                    // Our chunks are separated by 2 bits to prevent overlap.
                    for _ in 0..4 {
                        gen = gen.double(&tmp_params);
                    }
                }
                pedersen_circuit_generators.push(windows);
            }

            tmp_params.pedersen_circuit_generators = pedersen_circuit_generators;
        }

        // Create the 3-bit window table lookups for fixed-base
        // exp of each base in the protocol.
        {
            let mut fixed_base_circuit_generators = vec![];

            for mut gen in tmp_params.fixed_base_generators.iter().cloned() {
                let mut windows = vec![];
                for _ in 0..tmp_params.fixed_base_chunks_per_generator() {
                    let mut coeffs = vec![(Fr::zero(), Fr::one())];
                    let mut g = gen.clone();
                    for _ in 0..7 {
                        coeffs.push(g.into_xy());
                        g = g.add(&gen, &tmp_params);
                    }
                    windows.push(coeffs);

                    // gen = gen * 8
                    gen = g;
                }
                fixed_base_circuit_generators.push(windows);
            }

            tmp_params.fixed_base_circuit_generators = fixed_base_circuit_generators;
        }

        Ok(tmp_params)
    }
}

/// Determines whether or not we have to write/overwrite the exp-table file at `exp_table_path`.
///
/// If the exp-table file at `exp_table_path` exists and has the correct metadata (number of
/// segments and window-size) that is desired for these Pedersen Hash params, we return `false` to
/// indicate that no further exp-table generation needs to take place.
///
/// If the file doesn't exist, is not long enough (the number of segments in the exp-table file is
/// fewer than the number of segments required by the Pedersen Hash params), or the window-size in
/// the file differs from that being requested, we return `true` to indicate that a new exp-table
/// must be generated and written.
fn must_write_exp_table(
    n_segments_required: usize,
    window_size_required: u32,
    exp_table_path: &str,
) -> bool {
    let mut file = match File::open(exp_table_path) {
        Ok(file) => file,
        _ => return true,
    };

    let n_segments_in_file = {
        let mut buf = [0u8; 4];
        if file.read_exact(&mut buf).is_err() {
            return true;
        }
        u32::from_le_bytes(buf) as usize
    };

    if n_segments_in_file < n_segments_required {
        return true;
    }

    let window_size_in_file = {
        file.seek(SeekFrom::Start(4)).unwrap();
        let mut buf = [0u8; 4];
        if file.read_exact(&mut buf).is_err() {
            return true;
        }
        u32::from_le_bytes(buf)
    };

    window_size_in_file != window_size_required
}

/// Create the exp-table for the provided set of generators.
fn create_exp_table_for_generators(
    generators: &[edwards::Point<Bls12, PrimeOrder>],
    window_size: u32,
) -> Vec<Vec<Vec<edwards::Point<Bls12, PrimeOrder>>>> {
    let mut exp_table = Vec::with_capacity(generators.len());
    let n_windows_per_segment = (fs::Fs::NUM_BITS as f32 / window_size as f32).ceil() as usize;

    for g in generators.iter() {
        let mut g = g.clone();
        let mut segment_windows = Vec::with_capacity(n_windows_per_segment);

        let mut num_bits = 0;
        while num_bits <= fs::Fs::NUM_BITS {
            let mut window_values = Vec::with_capacity(1 << window_size);
            let mut base = edwards::Point::zero();

            for _ in 0..(1 << window_size) {
                window_values.push(base.clone());
                base = base.add(&g, &JUBJUB_BLS12_PARAMS_WITH_EDWARDS_D);
            }

            segment_windows.push(window_values);
            num_bits += window_size;

            for _ in 0..window_size {
                g = g.double(&JUBJUB_BLS12_PARAMS_WITH_EDWARDS_D);
            }
        }

        exp_table.push(segment_windows);
    }

    exp_table
}

/// Given a path to an exp-table file, this function creates any directories in that path that do
/// not exist.
fn create_exp_table_path_if_dne(path: &str) -> io::Result<()> {
    let path = Path::new(path);
    let dirs = path.parent().expect("invalid exp-table path");
    let path_is_just_file_name = dirs.to_str().unwrap().is_empty();

    if path_is_just_file_name {
        Ok(())
    } else {
        create_dir_all(dirs)
    }
}

/// Generates the Pedersen Hash parameter's exp-table in chunks of segments, writing each chunk of
/// the exp-table to the file at `exp_table_path`. This function generates the exp-table in such a
/// way that there is never more than `MAX_EXP_TABLE_SEGMENTS_IN_MEMORY` segments worth of the
/// exp-table in memory at a given time.
///
/// Returns an `Err` if we failed to create `exp_table_path` (create the directories in the path
/// that do not exist) or if we fail to create the exp-table file.
fn write_exp_table_to_file(
    generators: &[edwards::Point<Bls12, PrimeOrder>],
    window_size: u32,
    exp_table_path: &str,
) -> io::Result<()> {
    create_exp_table_path_if_dne(exp_table_path)?;

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(exp_table_path)?;

    // The first 8 bytes of the exp-table file are the number of segments in the exp-table (4 bytes)
    // and the window-size for this exp-table (4 bytes).
    let n_segments = generators.len();
    let n_segments_as_bytes = (n_segments as u32).to_le_bytes();
    let window_size_as_bytes = window_size.to_le_bytes();
    file.write_all(&n_segments_as_bytes)
        .expect("failed to write n_segments");
    file.write_all(&window_size_as_bytes)
        .expect("failed to write window_size");

    let mut n_segments_written = 0;

    // Generate and write the exp-table in chunks.
    while n_segments_written < n_segments {
        let n_segments_remaining = n_segments - n_segments_written;

        // The number of segments of the exp-table being generated in this chunk. It is possible for
        // the last chunk to contain fewer than `MAX_EXP_TABLE_SEGMENTS_IN_MEMORY` number of
        // segments of the exp-table.
        let n_segments_in_range = if n_segments_remaining > MAX_EXP_TABLE_SEGMENTS_IN_MEMORY {
            MAX_EXP_TABLE_SEGMENTS_IN_MEMORY
        } else {
            n_segments_remaining
        };

        // Calculate the range of segment indices for this chunk, create the generators
        // corresponding to each segment index, then create the exp-table corresponding to each of
        // the generators.
        let first_segment = n_segments_written;
        let stop_at_segment = first_segment + n_segments_in_range;
        let generators = &generators[first_segment..stop_at_segment];
        let exp_table = create_exp_table_for_generators(generators, window_size);

        // For each segment in this chunk, compress the segment's exp-table values (Edward points)
        // in parallel. Collect into a vector then write to the exp-table file to reduce the number
        // of file I/O operations.
        for segment_table in exp_table {
            let segment_table_bytes: Vec<u8> = segment_table
                .par_iter()
                .flat_map(|window| {
                    window
                        .par_iter()
                        .flat_map(|point| point.compress().as_bytes())
                        .collect::<Vec<u8>>()
                })
                .collect();

            file.write_all(&segment_table_bytes)
                .expect("failed to write segment table bytes");
        }

        n_segments_written += n_segments_in_range;
    }

    Ok(())
}

/// Loads into memory a range of the exp-table file. The range of the exp-table file that is read is
/// given by the range of segment indices: `first_segment..first_segment + n_segments`.
///
/// This function can return less than `n_segments` worth of the exp-table if `first_segment +
/// n_segments` exceeds the length of the exp-table file. If `first_segment` exceeds the number of
/// segments in the exp-table file an empty vector is returned. It is the user's responsibility to
/// check the number of exp-table segments returned from this function.
///
/// Returns an `Err` if there is no file at `exp_table_path` or the file cannot be read.
pub fn read_exp_table_range(
    first_segment: usize,
    n_segments: usize,
    exp_table_path: &str,
) -> io::Result<Vec<Vec<Vec<edwards::Point<Bls12, PrimeOrder>>>>> {
    let file = File::open(exp_table_path)?;

    // We wrap the exp-table file in `BufReader` to lessen the cost of many file reads.
    let mut reader = BufReader::new(file);

    // We skip the first 4 bytes in the file because those bytes encode the number of segments in
    // the file (which is a value that we will not use in this function).
    reader.seek(SeekFrom::Start(4)).unwrap();

    let window_size = {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .expect("failed to read bytes 4-8 from the exp-table file");
        u32::from_le_bytes(buf)
    };

    // Calculate the number of bytes we have to seek forward per segment in the exp-table file.
    let n_windows_per_segment = {
        let n_windows_per_segment = fs::Fs::NUM_BITS as f32 / window_size as f32;
        if n_windows_per_segment.fract() == 0.0 {
            (n_windows_per_segment as usize) + 1
        } else {
            n_windows_per_segment.ceil() as usize
        }
    };
    let n_points_per_window = 2usize.pow(window_size);
    let n_bytes_per_point = size_of::<Fr>();
    let n_bytes_per_segment = n_windows_per_segment * n_points_per_window * n_bytes_per_point;

    // Seek to the first byte in the exp-table that we want to read.
    let n_exp_table_bytes_to_skip = n_bytes_per_segment as i64 * first_segment as i64;
    reader
        .seek(SeekFrom::Current(n_exp_table_bytes_to_skip))
        .unwrap();

    // TODO: parallelize exp-reads - start by trying to read one segment per thread if `n_segments`
    // is small.

    // Read in the desired range of the exp-table.
    let mut exp_table = vec![];

    for _ in 0..n_segments {
        let mut segment_windows = vec![];
        for _ in 0..n_windows_per_segment {
            let mut window = vec![];
            for _ in 0..n_points_per_window {
                let read_point_res = edwards::Point::<Bls12, Unknown>::read(
                    &mut reader,
                    &JUBJUB_BLS12_PARAMS_WITH_EDWARDS_D,
                );

                let point = match read_point_res {
                    Ok(point) => point
                        .as_prime_order(&JUBJUB_BLS12_PARAMS_WITH_EDWARDS_D)
                        .expect("point read from exp-table does not belong to subgroup"),
                    Err(err) => {
                        match err.kind() {
                            io::ErrorKind::UnexpectedEof => {
                                // If we have reached the end of the file and have successfully read
                                // the last segment's table, return the exp-table. Otherwise, if we
                                // have reached the end of the exp-table file in the middle of a
                                // segment's table, panic.
                                if segment_windows.is_empty() && window.is_empty() {
                                    return Ok(exp_table);
                                }
                                panic!("exp-table file ends in the middle of a segment's table");
                            }
                            _ => {
                                panic!("point read from exp-table does not exist on curve");
                            }
                        }
                    }
                };

                window.push(point);
            }
            segment_windows.push(window);
        }
        exp_table.push(segment_windows);
    }

    Ok(exp_table)
}

#[test]
fn test_jubjub_bls12() {
    let params = JubjubBls12::new();

    tests::test_suite::<Bls12>(&params);

    let test_repr = hex!("9d12b88b08dcbef8a11ee0712d94cb236ee2f4ca17317075bfafc82ce3139d31");
    let p = edwards::Point::<Bls12, _>::read(&test_repr[..], &params).unwrap();
    let q = edwards::Point::<Bls12, _>::get_for_y(
        Fr::from_str(
            "22440861827555040311190986994816762244378363690614952020532787748720529117853",
        )
        .unwrap(),
        false,
        &params,
    )
    .unwrap();

    assert!(p == q);

    // Same thing, but sign bit set
    let test_repr = hex!("9d12b88b08dcbef8a11ee0712d94cb236ee2f4ca17317075bfafc82ce3139db1");
    let p = edwards::Point::<Bls12, _>::read(&test_repr[..], &params).unwrap();
    let q = edwards::Point::<Bls12, _>::get_for_y(
        Fr::from_str(
            "22440861827555040311190986994816762244378363690614952020532787748720529117853",
        )
        .unwrap(),
        true,
        &params,
    )
    .unwrap();

    assert!(p == q);
}
