// Specification:
// https://github.com/safetensors/safetensors

use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let safetensors_path = "../../gpt2/model.safetensors";
    let mut file = std::fs::File::open(safetensors_path)?;

    let mut size_of_header_buffer = [0u8; 8];
    file.read_exact(&mut size_of_header_buffer)?;
    let size_of_header = usize::from_le_bytes(size_of_header_buffer);
    println!("size_of_header: {size_of_header}");

    let mut header_buffer = vec!(0u8; size_of_header);
    file.read_exact(&mut header_buffer)?;
    let header = str::from_utf8(&header_buffer)?;
    println!("header: {header}");

    Ok(())
}
