// Specification:
// https://github.com/safetensors/safetensors

use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let safetensors_path = "../../gpt2/model.safetensors";
    let mut file = std::fs::File::open(safetensors_path)?;

    let mut size_of_raw_header_buffer = [0u8; 8];
    file.read_exact(&mut size_of_raw_header_buffer)?;
    let size_of_raw_header = usize::from_le_bytes(size_of_raw_header_buffer);
    println!("size_of_raw_header: {size_of_raw_header}");

    let mut raw_header_buffer = vec!(0u8; size_of_raw_header);
    file.read_exact(&mut raw_header_buffer)?;
    let raw_header = str::from_utf8(&raw_header_buffer)?;
    // println!("raw_header: {raw_header:?}");

    // Crate json:
    // https://docs.rs/json/latest/json/index.html

    let header = json::parse(raw_header)?;
    // println!("header: {header:?}");

    for (tensor_name, tensor_info) in header.entries() {
        println!("{tensor_name}");
        for (key, value) in tensor_info.entries() {
            match key {
                "dtype" => println!("    dtype: {value}"),
                "shape" => {
                    let mut shape: Vec<i32> = Vec::new();
                    for member in value.members() {
                        shape.push(member.as_i32().unwrap());
                    }
                    println!("    shape: {shape:?}");
                },
                "data_offsets" => {
                    let mut data_offsets: Vec<i32> = Vec::new();
                    for member in value.members() {
                        data_offsets.push(member.as_i32().unwrap());
                    }
                    println!("    data_offsets: {data_offsets:?}");
                },
                _ => {},
            }
        }
    }
    // JSON part done!

    let mut byte_buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut byte_buffer)?;
    println!("{}", byte_buffer.len());

    Ok(())
}
