// I still haven't implemented a tokenizer.
// Currently, I type token ids directly to debug my inference engine and that's tedious.
// Let's implement a tokenizer!

// I found a nice blog article.
// https://sebastianraschka.com/blog/2025/bpe-from-scratch.html

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    let token_to_id: HashMap<String, usize> = {
        let path = &format!("{}/vocab.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
    };

    println!("{token_to_id:?}");

    Ok(())
}
