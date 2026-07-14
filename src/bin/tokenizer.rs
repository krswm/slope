// I still haven't implemented a tokenizer.
// Currently, I type token ids directly to debug my inference engine and that's tedious.
// Let's implement a tokenizer!

// I found a nice blog article.
// https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    let token_to_id: HashMap<String, usize> = {
        let path = &format!("{}/vocab.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
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
            let token0 = split.next().unwrap();
            let token1 = split.next().unwrap();
            let id0 = &token_to_id[token0];
            let id1 = &token_to_id[token1];

            ranks.insert((*id0, *id1), rank);
            rank += 1;
        }
        ranks
    };

    // Token IDs
    let ids = {
        let input = &args[2];

        let mut raw_tokens = Vec::new();
        for (i_line, line) in input.split("\n").enumerate() {
            if i_line >= 1 {
                raw_tokens.push("\n".to_string());
            }

            for (i_word, word) in line.split(" ").enumerate() {
                if i_word == 0 && word != "" {
                    raw_tokens.push(word.to_string());
                } else if i_word >= 1 {
                    let mut raw_token = " ".to_string();
                    raw_token.push_str(word);
                    raw_tokens.push(raw_token);
                }
            }
        }

        let tokens: Vec<String> = raw_tokens
            .iter()
            .map(|raw_token| encode_unique_encoding(raw_token))
            .collect();

        let mut ids = Vec::new();
        for token in tokens.iter() {
            if token_to_id.contains_key(token) {
                ids.push(&token_to_id[token]);
            } else {
                todo!("every in spanish");
            }
        }

        println!("{tokens:?}");
        println!("{ids:?}");
    };

    Ok(())
}

fn encode_unique_encoding(text: &str) -> String {
    text.bytes()
        .map(|b| {
            let x = b as u32;
            TryInto::<char>::try_into(
                (match b {
                    0x00..=0x20 => x + 0x0100, // 0x0100..=0x0120
                    0x21..=0x7E => x,          // 0x0021..=0x007E
                    0x7F..=0xA0 => x + 0x00A2, // 0x0121..=0x0142
                    0xA1..=0xAC => x,          // 0x00A1..=0x00AC
                    0xAD => 0xAD,              // 0x0143
                    0xAE..=0xFF => x,          // 0x00AE..=0x00FF
                }),
            )
            .unwrap()
        })
        .collect()
}
