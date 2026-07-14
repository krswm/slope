// GPT-2 Inference with tenferro
// Copyright (C) 2026  Kurosawa Mutsumi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
            .map(|(key, value)| (*value, key.clone()))
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

    let config = {
        let path = &format!("{}/config.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let conf: HashMap<String, Value> = serde_json::from_reader(reader)?;
        transformer::Config {
            n_ctx: conf["n_ctx"].as_u64().unwrap() as usize,
            n_embd: conf["n_embd"].as_u64().unwrap() as usize,
            n_head: conf["n_head"].as_u64().unwrap() as usize,
            n_layer: conf["n_layer"].as_u64().unwrap() as usize,
            vocab_size: conf["vocab_size"].as_u64().unwrap() as usize,
        }
    };

    // ==== Tokenization ====

    let mut ids = tokenizer::tokenize(&token_to_id, &ranks, &args[2])?;

    println!();
    print!("\x1b[1;90m{}\x1b[22;39m", &args[2]);
    std::io::stdout().flush()?;

    // ==== Inference ====

    // The transposition of wte_weight is expensive since it usually contains millions of parameters.
    // Transpose it once and reuse it.
    let mut backend = CpuBackend::new();
    let wte_weight = &tensors["wte.weight"];
    if wte_weight.shape() != [config.vocab_size, config.n_embd] {
        return Err("tensor has unexpected shape".into());
    }
    let transposed_wte_weight = wte_weight.transpose(&[1, 0], &mut backend)?;

    let mut utf8_buffer = Vec::new();
    loop {
        match generate_next_id(
            &tensors,
            &transposed_wte_weight,
            &config,
            &ids,
            &mut backend,
        ) {
            Ok(next_id) => {
                ids.push(next_id);

                let decoded =
                    tokenizer::decode_unique_encoding(&id_to_token[&next_id], &mut utf8_buffer);
                print!("\x1b[1m{decoded}\x1b[22m");
                std::io::stdout().flush()?;
            }
            Err(err) => {
                println!();
                return Err(err);
            }
        };
    }
}

fn generate_next_id(
    tensors: &HashMap<String, TypedTensor<f32>>,
    transposed_wte_weight: &TypedTensor<f32>,
    config: &transformer::Config,
    ids: &Vec<usize>,
    backend: &mut CpuBackend,
) -> Result<usize, Box<dyn Error>> {
    let output = transformer::transform(tensors, transposed_wte_weight, config, ids, backend)?;

    // Greedy sampling: Choose the token with the highest probability.
    let mut best_prob = f32::NEG_INFINITY;
    let mut next_id = 0;
    for col in 0..config.vocab_size {
        let prob = *output.get(&[ids.len() - 1, col])?;
        if prob > best_prob {
            best_prob = prob;
            next_id = col;
        }
    }

    Ok(next_id)
}
