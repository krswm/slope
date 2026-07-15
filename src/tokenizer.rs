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

/// Tokenize an input with the BPE algorithm.
pub fn tokenize(
    token_to_id: &HashMap<String, usize>,
    ranks: &HashMap<(String, String), u32>,
    input: &str,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let raw_tokens = {
        let mut raw_tokens = Vec::new();
        for (i_line, line) in input.split("\n").enumerate() {
            if i_line >= 1 {
                raw_tokens.push(String::from("\n"));
            }

            for (i_word, word) in line.split(" ").enumerate() {
                if i_word == 0 && !word.is_empty() {
                    raw_tokens.push(word.to_string());
                } else if i_word >= 1 {
                    raw_tokens.push(format!(" {word}"));
                }
            }
        }
        raw_tokens
    };

    let tokens: Vec<String> = raw_tokens
        .iter()
        .map(|raw_token| encode_unique_encoding(raw_token))
        .collect();

    // Token IDs
    let ids = {
        let mut ids = Vec::new();
        for token in tokens.iter() {
            if token_to_id.contains_key(token) {
                ids.push(token_to_id[token]);
            } else {
                // ==== Merge ====

                let mut symbols: Vec<String> = token.chars().map(|x| x.to_string()).collect();

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

                    let mut best_rank = u32::MAX;
                    let mut best_i_pair = usize::MAX;
                    for (i_pair, pair) in pairs.iter().enumerate() {
                        if ranks.contains_key(pair) && ranks[pair] < best_rank {
                            best_rank = ranks[pair];
                            best_i_pair = i_pair;
                        }
                    }
                    if best_i_pair == usize::MAX {
                        break;
                    }

                    symbols[best_i_pair] =
                        format!("{}{}", symbols[best_i_pair], symbols[best_i_pair + 1]);
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

// GPT-2 has a unique encoding.
// e.g.: 'Ġ' (U+0120) → 0x20

fn encode_unique_encoding(text: &str) -> String {
    text.bytes()
        .map(|x| {
            let y = x as u32;
            TryInto::<char>::try_into(match x {
                0x00..=0x20 => y + 0x0100, // 0x0100..=0x0120
                0x21..=0x7E => y,          // 0x0021..=0x007E
                0x7F..=0xA0 => y + 0x00A2, // 0x0121..=0x0142
                0xA1..=0xAC => y,          // 0x00A1..=0x00AC
                0xAD => 0x0143,            // 0x0143 (0xAD is SOFT HYPHEN)
                0xAE..=0xFF => y,          // 0x00AE..=0x00FF
            })
            .unwrap()
        })
        .collect()
}

pub fn decode_unique_encoding(text: &str, utf8_buffer: &mut Vec<u8>) -> String {
    let new_buffer: Vec<u8> = text
        .chars()
        .map(|x| {
            let y = x as u32;
            (match y {
                0x0100..=0x0120 => y - 0x0100, // 0x00..=0x20
                0x0021..=0x007E => y,          // 0x21..=0x7E
                0x0121..=0x0142 => y - 0x00A2, // 0x7F..=0xA0
                0x00A1..=0x00AC => y,          // 0xA1..=0xAC
                0x0143 => 0xAD,                // 0xAD (0xAD is SOFT HYPHEN)
                0x00AE..=0x00FF => y,          // 0xAE..=0xFF
                _ => 0,
            }) as u8
        })
        .collect();

    // A token may contain only a part of UTF-8 sequence.
    // Decode it incrementally.

    let mut buffer = utf8_buffer.clone();
    buffer.extend(new_buffer);
    utf8_buffer.clear();

    let mut decoded = String::new();
    loop {
        match std::str::from_utf8(&buffer) {
            Ok(valid) => {
                decoded.push_str(valid);
                return decoded;
            }
            Err(err) => {
                let (valid, remaining) = buffer.split_at(err.valid_up_to());
                decoded.push_str(std::str::from_utf8(valid).unwrap());

                if let Some(invalid_len) = err.error_len() {
                    decoded.push(char::REPLACEMENT_CHARACTER);
                    buffer = remaining[invalid_len..].to_vec();
                } else {
                    *utf8_buffer = remaining.to_vec();
                    return decoded;
                }
            }
        }
    }
}
