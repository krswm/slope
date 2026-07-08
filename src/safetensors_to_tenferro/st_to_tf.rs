// Specification:
// https://github.com/safetensors/safetensors

use std::io::Read;
use std::collections::HashMap;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub fn st_to_tf(safetensors_path: &str) -> Result<HashMap<String, TypedTensor<f32>>, Box<dyn std::error::Error>> {
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

    let mut tensors = HashMap::new();

    for (tensor_name, tensor_info) in header.entries() {
        print!("{tensor_name}");

        let mut begin = 0usize;
        let mut end;

        let mut dtype = "";

        let mut shape0 = 0usize;
        let mut shape1 = 0usize;
        let mut size = 0usize;

        for (key, value) in tensor_info.entries() {
            match key {
                "dtype" => {
                    // print!(" dtype: {value}");

                    dtype = value.as_str().unwrap();
                },
                "shape" => {
                    let mut shape = Vec::new();
                    for member in value.members() {
                        shape.push(member.as_usize().unwrap());
                    }
                    print!(" shape: {shape:?}");

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
                },
                "data_offsets" => {
                    let mut data_offsets = Vec::new();
                    for member in value.members() {
                        data_offsets.push(member.as_usize().unwrap());
                    }
                    // print!(" data_offsets: {data_offsets:?}");

                    begin = *data_offsets.get(0).unwrap();
                    end = *data_offsets.get(1).unwrap();

                    // print!(" begin: {begin}, end: {end}");
                },
                _ => {},
            }
        }
        println!();

        if dtype == "F32" && shape0 != 0 {
            if shape1 == 0 {  // 1D
                let mut raw_tensor = Vec::new();

                for i in 0..size {
                    // F32 is 4 bytes long.
                    let b = begin + 4 * i;
                    let e = b + 4;

                    let f = f32::from_le_bytes(*byte_buffer[b..e].as_array::<4>().unwrap());

                    raw_tensor.push(f);
                }

                let tensor: TypedTensor<f32> = TypedTensor::<f32>::from_vec_col_major(vec![shape0], raw_tensor)?;  // 1D
            } else {  // 2D
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

                for i in 0..size {
                    // F32 is 4 bytes long.
                    let b = begin + 4 * i;
                    let e = b + 4;

                    let f = f32::from_le_bytes(*byte_buffer[b..e].as_array::<4>().unwrap());

                    raw_tensor.push(f);
                }

                // I loaded the safetensor file to raw_tensor
                // Now raw_tensor is
                //
                // a b c d e f 

                let tensor_direct = TypedTensor::<f32>::from_vec_col_major(vec![shape1, shape0], raw_tensor)?;  // 2D

                // I directly fed raw_tensor to tenferro
                // Now tensor_direct is
                //
                // a d
                // b e
                // c f
                //
                // num of rows    = 3 = shape_1
                // num of columns = 2 = shape_0
                //
                // Of course it is not alright

                let mut backend = tenferro_cpu::CpuBackend::new();
                let tensor = tensor_direct.transpose(&[1, 0], &mut backend).unwrap();

                assert_eq!(tensor.shape(), vec![shape0, shape1]);

                // Solution: matrix transposition
                //
                //    T
                // a d    a b c
                // b e  = d e f
                // c f
                //
                // Now tensor is
                // num of rows = 2 = shape0
                // num of cols = 3 = shape1

                tensors.insert(tensor_name.to_string(), tensor);
            }
        }
    }
    // JSON part done!

    Ok(tensors)
}
