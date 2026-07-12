use std::io::Write;

pub mod loader;
pub mod transformer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let safetensors_path = &format!("{}/model.safetensors", &args[1]);
    let vocab_path = &format!("{}/vocab.json", &args[1]);

    let vocab_raw_json = std::fs::read_to_string(vocab_path)?;
    let vocab = json::parse(&vocab_raw_json)?;

    let mut id_to_token = std::collections::HashMap::<usize, &str>::new();
    for (token, id) in vocab.entries() {
        id_to_token.insert(id.as_usize().unwrap(), token);
    }

    let tensors = loader::load_safetensors(safetensors_path)?;

    let mut ids = args[2..]
        .into_iter()
        .map(|id| id.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();

    let n_vocab = 50257; // TODO: Don't hardcode this!

    /*
    for id in &ids {
        print!(
            "\x1b[1m{}\x1b[22m",
            replace_characters(id_to_token.get(&id).unwrap())
        );
        std::io::stdout().flush().unwrap();
    }
    */

    for _ in 0..1 {
        let a = transformer::transform(&tensors, &ids)?;

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
