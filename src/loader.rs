use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};

use serde::Deserialize;
use serde_json::Value;
use tenferro_runtime::TypedTensor;

#[derive(Deserialize, Debug)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: Vec<usize>,
}

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

    let header: HashMap<String, Value> = {
        let mut buffer = vec![0; size_of_header];
        file.read_exact(&mut buffer)?;
        println!("{:?}", str::from_utf8(&buffer));
        serde_json::from_slice(&buffer)?
    };

    let byte_buffer = {
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        buffer
    };

    let tensors = {
        let mut tensors = HashMap::new();

        print!("\x1b[90mLoading Safetensors…\x1b[39m ");
        std::io::stdout().flush().unwrap();

        for (tensor_name, raw_tensor_info) in header.into_iter() {
            if tensor_name == "__metadata__" {
                continue;
            }

            let tensor_info: TensorInfo = serde_json::from_value(raw_tensor_info)?;
            
            print!("\x1b[100m \x1b[49m");
            std::io::stdout().flush().unwrap();

            let begin = tensor_info.data_offsets[0];
            let dtype = tensor_info.dtype;
            let (shape0, shape1, size) = if tensor_info.shape.len() == 2 {
                (
                    tensor_info.shape[0],
                    tensor_info.shape[1],
                    tensor_info.shape[0] * tensor_info.shape[1],
                )
            } else {
                (tensor_info.shape[0], 0usize, tensor_info.shape[0])
            };

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
        println!(" \x1b[90mDone.\x1b[39m");

        tensors
    };

    Ok(tensors)
}
