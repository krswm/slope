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
        println!("{input:?}");

        let mut tokens = Vec::new();
        for (i_line, line) in input.split("\n").enumerate() {
            if i_line >= 1 {
                tokens.push("\n".to_string());
            }

            for (i_word, word) in line.split(" ").enumerate() {
                if word != "" {
                    tokens.push(
                        if i_word == 0 { word.to_string() } else { let mut token = " ".to_string(); token.push_str(word); token.clone() }
                    );
                }
            }
        }

        println!("{tokens:?}");
    };

    Ok(())
}
