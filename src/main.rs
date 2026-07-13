use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Write};

use serde_json::Value;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub mod loader;
pub mod transformer;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    let tensors = {
        let path = &format!("{}/model.safetensors", &args[1]);
        loader::load_safetensors(path)?
    };

    let id_to_token: HashMap<usize, String> = {
        let path = &format!("{}/vocab.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let token_to_id: HashMap<String, usize> = serde_json::from_reader(reader)?;
        token_to_id
            .iter()
            .map(|(key, value)| (value.clone(), key.clone()))
            .collect()
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

    // Token IDs
    let mut ids: Vec<usize> = args[2..]
        .into_iter()
        .map(|id| id.parse().unwrap())
        .collect();

    let mut backend = CpuBackend::new();
    let wte_weight = &tensors["wte.weight"];
    if wte_weight.shape() != &[vocab_size, n_embd] {
        return Err("tensor has unexpected shape".into());
    }
    let transposed_wte_weight = wte_weight.transpose(&[1, 0], &mut backend)?;

    println!();
    for id in &ids {
        print!(
            "\x1b[1;90m{}\x1b[22;39m",
            decode_unique_encoding(&id_to_token[&id])
        );
    }
    std::io::stdout().flush()?;

    for _ in 0..100 {
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

        print!(
            "\x1b[1m{}\x1b[22m",
            decode_unique_encoding(&id_to_token[&next_id])
        );
        std::io::stdout().flush()?;
    }
    println!();

    Ok(())
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

fn decode_unique_encoding(text: &str) -> String {
    // GPT-2 has a unique encoding.
    // e.g.: 'Ġ' (U+0120) → ' ' (U+0020)

    text.chars()
        .map(|c| {
            if c as u32 >= 0x100 {
                char::from_u32(c as u32 - 0x100).unwrap()
            } else {
                c
            }
        })
        .collect()
}
