use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};

use serde_json::Value;

pub mod loader;
pub mod transformer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let safetensors_path = &format!("{}/model.safetensors", &args[1]);
    let vocab_path = &format!("{}/vocab.json", &args[1]);
    let config_path = &format!("{}/config.json", &args[1]);

    let vocab_raw_json = std::fs::read_to_string(vocab_path)?;
    let vocab = json::parse(&vocab_raw_json)?;

    let mut id_to_token = std::collections::HashMap::<usize, &str>::new();
    for (token, id) in vocab.entries() {
        id_to_token.insert(id.as_usize().unwrap(), token);
    }

    let (n_ctx, n_embd, n_head, n_layer, vocab_size) = {
        let file = File::open(config_path)?;
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

    let tensors = loader::load_safetensors(safetensors_path)?;

    let mut ids = args[2..]
        .into_iter()
        .map(|id| id.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();

    let n_vocab = 50257; // TODO: Don't hardcode this!

    for id in &ids {
        print!(
            "\x1b[1m{}\x1b[22m",
            replace_characters(id_to_token.get(&id).unwrap())
        );
        std::io::stdout().flush().unwrap();
    }

    for _ in 0..10 {
        let a = transformer::transform(&tensors, n_ctx, n_embd, n_head, n_layer, vocab_size, &ids)?;

        let mut next_id = 0;
        let mut max = -1.0e12f32; // I'll do greedy sampling
        for col in 0..n_vocab {
            let b = *a.get(&[ids.len() - 1, col])?;
            if b > max {
                max = b;
                next_id = col;
            }
        }

        print!(
            "\x1b[1;35m{}\x1b[22;39m",
            replace_characters(id_to_token.get(&next_id).unwrap())
        );
        std::io::stdout().flush().unwrap();

        ids.push(next_id);
    }
    println!();

    // Needs refactoring!

    Ok(())
}

fn replace_characters(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch as u32 >= 0x100 {
                char::from_u32(ch as u32 - 0x100).unwrap()
            } else {
                ch
            }
        })
        .collect()
}
