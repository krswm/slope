use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use serde_json::Value;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub mod loader;
pub mod tokenizer;
pub mod transformer;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    // ==== Loading Files ====

    let tensors = {
        let path = &format!("{}/model.safetensors", &args[1]);
        loader::load_safetensors(path)?
    };

    let (token_to_id, id_to_token) = {
        let path = &format!("{}/vocab.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let token_to_id: HashMap<String, usize> = serde_json::from_reader(reader)?;
        let id_to_token: HashMap<usize, String> = token_to_id
            .iter()
            .map(|(key, value)| (value.clone(), key.clone()))
            .collect();
        (token_to_id, id_to_token)
    };

    let ranks = {
        let path = &format!("{}/merges.txt", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut ranks = HashMap::new();
        let mut rank = 0u32;
        for line in reader.lines().map_while(Result::ok) {
            // Skip a comment line.
            if line.starts_with("#") {
                continue;
            }

            let mut split = line.split(" ");
            let token0 = split.next().unwrap().to_string();
            let token1 = split.next().unwrap().to_string();

            ranks.insert((token0, token1), rank);
            rank += 1;
        }
        ranks
    };

    let (n_ctx, n_embd, n_head, n_layer, vocab_size) = {
        let path = &format!("{}/config.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let config: HashMap<String, Value> = serde_json::from_reader(reader)?;
        (
            config["n_ctx"].as_u64().unwrap() as usize,
            config["n_embd"].as_u64().unwrap() as usize,
            config["n_head"].as_u64().unwrap() as usize,
            config["n_layer"].as_u64().unwrap() as usize,
            config["vocab_size"].as_u64().unwrap() as usize,
        )
    };

    let mut backend = CpuBackend::new();
    let wte_weight = &tensors["wte.weight"];
    if wte_weight.shape() != &[vocab_size, n_embd] {
        return Err("tensor has unexpected shape".into());
    }
    let transposed_wte_weight = wte_weight.transpose(&[1, 0], &mut backend)?;

    // ==== Tokenization ====

    let mut ids = tokenizer::tokenize(&token_to_id, &ranks, &args[2])?;

    println!();
    print!("\x1b[1;90m{}\x1b[22;39m", &args[2]);
    std::io::stdout().flush()?;

    // ==== Inference ====

    let mut utf8_buffer = Vec::new();
    loop {
        let next_id = match generate_next_id(
            &tensors,
            &transposed_wte_weight,
            n_ctx,
            n_embd,
            n_head,
            n_layer,
            vocab_size,
            &ids,
            &mut backend,
        ) {
            Ok(next_id) => next_id,
            Err(err) => {
                println!();
                return Err(err);
            }
        };

        ids.push(next_id);

        let decoded = tokenizer::decode_unique_encoding(&id_to_token[&next_id], &mut utf8_buffer);
        print!("\x1b[1m{decoded}\x1b[22m");
        std::io::stdout().flush()?;
    }
}

fn generate_next_id(
    tensors: &HashMap<String, TypedTensor<f32>>,
    transposed_wte_weight: &TypedTensor<f32>,
    n_ctx: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    vocab_size: usize,
    ids: &Vec<usize>,
    backend: &mut CpuBackend,
) -> Result<usize, Box<dyn Error>> {
    let a = transformer::transform(
        &tensors,
        &transposed_wte_weight,
        n_ctx,
        n_embd,
        n_head,
        n_layer,
        vocab_size,
        &ids,
        backend,
    )?;

    // Greedy sampling: Choose the token with highest probability.
    let mut max = f32::NEG_INFINITY;
    let mut next_id = 0;
    for col in 0..vocab_size {
        let b = *a.get(&[ids.len() - 1, col])?;
        if b > max {
            max = b;
            next_id = col;
        }
    }

    Ok(next_id)
}
