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

    let mut byte_buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut byte_buffer)?;
    println!("{}", byte_buffer.len());

    // Crate json:
    // https://docs.rs/json/latest/json/index.html

    let header = json::parse(raw_header)?;
    // println!("header: {header:?}");

    let mut raw_tensors = std::collections::HashMap::new();

    for (tensor_name, tensor_info) in header.entries() {
        println!("{tensor_name}");

        let mut begin = 0usize;
        let mut end;

        let mut dtype = "";

        let mut shape0 = 0usize;
        let mut shape1 = 0usize;
        let mut size = 0usize;

        for (key, value) in tensor_info.entries() {
            match key {
                "dtype" => {
                    println!("    dtype: {value}");

                    dtype = value.as_str().unwrap();
                },
                "shape" => {
                    let mut shape = Vec::new();
                    for member in value.members() {
                        shape.push(member.as_usize().unwrap());
                    }
                    println!("    shape: {shape:?}");

                    if shape.len() == 2 {
                        shape0 = *shape.get(0).unwrap();
                        shape1 = *shape.get(1).unwrap();
                        size = shape0 * shape1;
                    } else if shape.len() == 1 {
                        shape0 = *shape.get(0).unwrap();
                        shape1 = 0usize;
                        size = shape0;
                    }

                    println!("    shape0: {shape0}, shape1: {shape1}, size: {size}");
                },
                "data_offsets" => {
                    let mut data_offsets = Vec::new();
                    for member in value.members() {
                        data_offsets.push(member.as_usize().unwrap());
                    }
                    println!("    data_offsets: {data_offsets:?}");

                    begin = *data_offsets.get(0).unwrap();
                    end = *data_offsets.get(1).unwrap();

                    println!("    begin: {begin}, end: {end}");
                },
                _ => {},
            }
        }

        if dtype == "F32" {
            let mut raw_tensor = Vec::new();

            for i in 0..size {
                // F32 is 4 bytes long.
                let b = begin + 4 * i;
                let e = b + 4;

                let f = f32::from_le_bytes(*byte_buffer[b..e].as_array::<4>().unwrap());

                raw_tensor.push(f);
            }

            raw_tensors.insert(tensor_name, raw_tensor);
        }
    }
    // JSON part done!

    Ok(())
}
