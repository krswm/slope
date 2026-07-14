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
        let lines = reader.lines();

        let mut ranks: HashMap<(usize, usize), u32> = HashMap::new();
        let mut rank = 0;
        for line in lines.map_while(Result::ok) {
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

    println!("{ranks:?}");

    Ok(())
}
