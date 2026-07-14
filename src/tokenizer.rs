use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Tokenize an input.
pub fn tokenize(input: &str) -> Result<Vec<usize>, Box<dyn Error>> {
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
            let token0 = split.next().unwrap().to_string();
            let token1 = split.next().unwrap().to_string();

            ranks.insert((token0, token1), rank);
            rank += 1;
        }
        ranks
    };

    // Token IDs
    let ids = {
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

        let mut ids: Vec<usize> = Vec::new();
        for token in tokens.iter() {
            if token_to_id.contains_key(token) {
                ids.push(token_to_id[token]);
            } else {
                let mut symbols: Vec<String> = token.chars().map(|c| c.to_string()).collect();

                while symbols.len() >= 2 {
                    let pairs = {
                        let mut pairs = Vec::with_capacity(symbols.len() - 1);
                        for i_char in 0..symbols.len() - 1 {
                            let token0 = symbols[i_char].clone();
                            let token1 = symbols[i_char + 1].clone();
                            pairs.push((token0, token1));
                        }
                        pairs
                    };

                    if pairs.len() == 0 {
                        break;
                    }

                    let mut best_i_pair = usize::MAX;
                    let mut best_rank = u32::MAX;
                    for (i_pair, pair) in pairs.iter().enumerate() {
                        if ranks.contains_key(pair) && ranks[pair] < best_rank {
                            best_i_pair = i_pair;
                            best_rank = ranks[pair];
                        }
                    }

                    if best_i_pair == usize::MAX {
                        break;
                    }

                    let mut best_pair = symbols[best_i_pair].clone();
                    best_pair.push_str(&symbols[best_i_pair + 1]);
                    symbols[best_i_pair] = best_pair;
                    symbols.remove(best_i_pair + 1);
                }

                for symbol in symbols {
                    ids.push(token_to_id[&symbol]);
                }
            }
        }

        ids
    };

    Ok(ids)
}

fn encode_unique_encoding(text: &str) -> String {
    text.bytes()
        .map(|b| {
            let x = b as u32;
            TryInto::<char>::try_into(match b {
                0x00..=0x20 => x + 0x0100, // 0x0100..=0x0120
                0x21..=0x7E => x,          // 0x0021..=0x007E
                0x7F..=0xA0 => x + 0x00A2, // 0x0121..=0x0142
                0xA1..=0xAC => x,          // 0x00A1..=0x00AC
                0xAD => 0xAD,              // 0x0143
                0xAE..=0xFF => x,          // 0x00AE..=0x00FF
            })
            .unwrap()
        })
        .collect()
}
