use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use tenferro_runtime::TypedTensor;

/// Load a Safetensors file and convert the tensors into `tenferro_runtime::TypedTensor<f32>`.
///
/// [Specification of the Safetensors file format](https://github.com/safetensors/safetensors#format)
pub fn load_safetensors(
    path_to_safetensors: &str,
) -> Result<HashMap<String, TypedTensor<f32>>, Box<dyn Error>> {
    let mut file = File::open(path_to_safetensors)?;

    let size_of_header = {
        let mut buffer = [0; 8];
        file.read_exact(&mut buffer)?;
        usize::from_le_bytes(buffer)
    };

    let byte_buffer = {
        let mut buffer = Vec::new();
        file.seek(SeekFrom::Start(8 + size_of_header as u64))?;
        file.read_to_end(&mut buffer)?;
        buffer
    };

    let mut raw_header_buffer = vec![0u8; size_of_header];
    file.seek(SeekFrom::Start(8))?;
    file.read_exact(&mut raw_header_buffer)?;
    let raw_header = str::from_utf8(&raw_header_buffer)?;

    // Crate json:
    // https://docs.rs/json/latest/json/index.html

    let header = json::parse(raw_header)?;
    // println!("header: {header:?}");

    let mut tensors = HashMap::new();

    print!("\x1b[37mLoading Safetensors... \x1b[39m");
    std::io::stdout().flush().unwrap();
    for (tensor_name, tensor_info) in header.entries() {
        // print!("{tensor_name}");
        print!("\x1b[47m \x1b[49m");
        std::io::stdout().flush().unwrap();

        let mut begin = 0usize;

        let mut dtype = "";

        let mut shape0 = 0usize;
        let mut shape1 = 0usize;
        let mut size = 0usize;

        for (key, value) in tensor_info.entries() {
            match key {
                "dtype" => {
                    // print!(" dtype: {value}");

                    dtype = value.as_str().unwrap();
                }
                "shape" => {
                    let mut shape = Vec::new();
                    for member in value.members() {
                        shape.push(member.as_usize().unwrap());
                    }
                    // print!(" shape: {shape:?}");

                    // The safetensor file contains
                    // 1D, 2D, and 4D tensors
                    // (for GPT-2 model file).
                    //
                    // The 4D tensors are just 1x1x1024x1024:
                    //
                    // 1 0 0 0 . 0
                    // 1 1 0 0 . 0
                    // 1 1 1 0 . 0
                    // 1 1 1 1 . 0
                    // . . . . . .
                    // 1 1 1 1 . 1
                    //
                    // (triangle matrix) and I don't need it
                    // because technically it can be re-created on the fly
                    //
                    // So I need to get only 1D and 2D.
                    //
                    // 1D is shape0 (shape1 set to 0 to signify it's 1D)
                    // 2D is shape0 x shape1

                    if shape.len() == 2 {
                        shape0 = *shape.get(0).unwrap();
                        shape1 = *shape.get(1).unwrap();
                        size = shape0 * shape1;
                    } else if shape.len() == 1 {
                        shape0 = *shape.get(0).unwrap();
                        shape1 = 0usize;
                        size = shape0;
                    }

                    // print!(" shape0: {shape0}, shape1: {shape1}, size: {size}");
                }
                "data_offsets" => {
                    let mut data_offsets = Vec::new();
                    for member in value.members() {
                        data_offsets.push(member.as_usize().unwrap());
                    }
                    // print!(" data_offsets: {data_offsets:?}");

                    begin = *data_offsets.get(0).unwrap();
                }
                _ => {}
            }
        }
        // println!();

        if dtype == "F32" && shape0 != 0 {
            if shape1 == 0 {
                // 1D
                let mut raw_tensor = Vec::new();

                for i in 0..size {
                    // F32 is 4 bytes long.
                    let b = begin + 4 * i;
                    let e = b + 4;

                    let f = f32::from_le_bytes(*byte_buffer[b..e].as_array::<4>().unwrap());

                    raw_tensor.push(f);
                }

                let tensor: TypedTensor<f32> =
                    TypedTensor::<f32>::from_vec_col_major(vec![shape0], raw_tensor)?; // 1D

                tensors.insert(tensor_name.to_string(), tensor);
            } else {
                // 2D
                // Safetensors is ROW-major
                // source: https://github.com/safetensors/safetensors#format
                //
                // 1 2 3
                // 4 5 6

                // tenferro is COLUMN-major
                // source: https://tensor4all.org/tenferro-rs/getting-started/pytorch-jax-mapping.html#column-major-storage
                //
                // 1 3 5
                // 2 4 6

                // Suppose We have a matrix
                //
                // a b c
                // d e f
                //
                // shape_0 = 2  (num of rows)
                // shape_1 = 3  (num of columns)

                // It is stored in Safetensors file as
                //
                // a b c d e f

                let mut raw_tensor = Vec::new();

                for col in 0..shape1 {
                    for row in 0..shape0 {
                        // F32 is 4 bytes long.
                        let b = begin + 4 * (row * shape1 + col); // Safetensors file is ROW-major!
                        let e = b + 4;

                        let f = f32::from_le_bytes(*byte_buffer[b..e].as_array::<4>().unwrap());

                        raw_tensor.push(f);
                    }
                }

                let tensor =
                    TypedTensor::<f32>::from_vec_col_major(vec![shape0, shape1], raw_tensor)?; // 2D

                tensors.insert(tensor_name.to_string(), tensor);
            }
        }
    }
    // JSON part done!
    println!();

    Ok(tensors)
}
