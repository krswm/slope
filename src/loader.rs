use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};

use serde::Deserialize;
use serde_json::Value;
use tenferro_runtime::TypedTensor;

#[derive(Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
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
        serde_json::from_slice(&buffer)?
    };

    let byte_buffer = {
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        buffer
    };

    let mut tensors = HashMap::new();

    print!("\x1b[90mLoading Safetensors…\x1b[39m ");
    std::io::stdout().flush()?;

    for (key, value) in header.into_iter() {
        // I do not use metadata in my inferenece engine.
        if key == "__metadata__" {
            continue;
        }

        let name = key;
        let info: Info = serde_json::from_value(value)?;

        // My inference engine uses only f32 tensors.
        if info.dtype != "F32" {
            continue;
        }

        // f32 is 4 bytes long.
        let size = 4 * info.shape.iter().product::<usize>();
        let begin = info.data_offsets[0];
        let end = info.data_offsets[0];
        if 4 * size < end - begin {
            return Err("tensor data smaller than tensor shape suggests".into());
        }

        let rowmaj = byte_buffer[begin..begin + size]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(*chunk.as_array::<4>().unwrap()))
            .collect();

        // ⎛a₁₁ a₁₂ a₁₃⎞
        // ⎝a₂₁ a₂₂ a₂₃⎠
        //
        // Safetensors uses row-major: a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃.
        // https://github.com/safetensors/safetensors#format
        //
        // tenferro uses column-major: a₁₁ a₂₁ a₁₂ a₂₂ a₁₃ a₂₃.
        // https://tensor4all.org/tenferro-rs/getting-started/pytorch-jax-mapping.html#column-major-storage

        // My inference engine uses only 1D and 2D tensors.
        match info.shape.len() {
            1 => {
                // Row-major and column-major are identical in 1D.
                let colmaj = rowmaj;

                let tensor = TypedTensor::<f32>::from_vec_col_major(info.shape, colmaj)?;
                tensors.insert(name.to_string(), tensor);
            }
            2 => {
                let mut colmaj = Vec::with_capacity(info.shape[0] * info.shape[1]);

                for col in 0..info.shape[1] {
                    for row in 0..info.shape[0] {
                        colmaj.push(rowmaj[row * info.shape[1] + col]);
                    }
                }

                let tensor = TypedTensor::<f32>::from_vec_col_major(info.shape, colmaj)?;
                tensors.insert(name, tensor);
            }
            _ => (),
        }

        print!("\x1b[100m \x1b[49m");
        std::io::stdout().flush()?;
    }
    println!(" \x1b[90mDone.\x1b[39m");

    Ok(tensors)
}
