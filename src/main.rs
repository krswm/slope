// // My first tenferro
// // https://tensor4all.org/tenferro-rs/getting-started/#first-cpu-program
// 
// // It seems I need this in order to a.matmul
// use tenferro_runtime::TensorOpsExt;
// 
// use tenferro_runtime::TypedTensor;
// 
// // What is Box?
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     let mut backend = tenferro_cpu::CpuBackend::new();
// 
//     let a = tenferro_runtime::Tensor::from_vec_col_major(
//         vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]
//     )?;
//     let b = tenferro_runtime::Tensor::from_vec_col_major(
//         vec![2, 2], vec![5.0, 7.0, 6.0, 8.0]
//     )?;
// 
//     // a @ b?
//     let c = a.matmul(&b, &mut backend)?;
// 
//     println!("{:?}", c.shape());
//     println!("{:?}", c.as_slice::<f64>().unwrap());
// 
//     println!("----");
// 
//     // I need transpose
// 
//     let d = tenferro_runtime::Tensor::from_vec_col_major(
//         vec![2, 3], vec![1.0, 3.0, 2.0, 4.0, 10.0, 20.0]
//     )?;
//     println!("{:?}", d.as_slice::<f64>().unwrap());
//     let e = d.transpose(&[1, 0], &mut backend)?;
//     println!("{:?}", e.as_slice::<f64>().unwrap());
// 
//     println!("----");
// 
//     // I need matrix addition
// 
//     // https://tensor4all.org/tenferro-rs/spec/operation-categories.html#elementwise-arithmetic-comparison-selection 
// 
//     let f = tenferro_runtime::Tensor::from_vec_col_major(
//         vec![2, 3], vec![1.0, 3.0, 2.0, 4.0, -1.0, -2.0]
//     )?;
//     let g = tenferro_runtime::Tensor::from_vec_col_major(
//         vec![2, 3], vec![0.1, 0.2, 0.2, 0.4, -0.1, -0.2]
//     )?;
// 
//     let h = f.add(&g, &mut backend)?;
//     println!("{:?}", h.as_slice::<f64>().unwrap());
// 
//     println!("----");
// 
//     // I need upper triangular matrix
// 
//     // https://tensor4all.org/tenferro-rs/spec/operation-categories.html#construction
// 
//     // I may have to use TypedTensor instead of Tensor
//     // since I will use only float32 tensors
// 
//     // T is type but what is R?
//     // https://tensor4all.org/tenferro-rs/api/tenferro_tensor/types/struct.TypedTensor.html#method.ones
// 
//     let i = TypedTensor::<f32>::ones(vec![2, 3])?;
//     println!("{:?}", i.host_data()?);
// 
//     // Eager? Traced?
// 
//     // let j = i.triu(&mut backend)?;
//     // println!("{:?}", j.host_data()?);
// 
//     Ok(())
// }

use tenferro_runtime::TypedTensor;

pub mod safetensors_to_tenferro;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let safetensors_path = "../gpt2/model.safetensors";  // FIXME later: Don't hardcode a path! Ask for a path instead.

    let tensors = safetensors_to_tenferro::st_to_tf::st_to_tf(safetensors_path)?;

    let ids = vec![40, 1842, 19617, 13];

    // // Input embedding // //

    let wte_weight: &TypedTensor<f32> = tensors.get("wte.weight").unwrap(); 

    // How to extract a single row in tenferro?
    // let x = wte_weight.get(&[2usize, :])?;
    // println!("{x:?}");
    // How???

    // OK do it manually
    let wte_weight_shape = wte_weight.shape();
    let n_ids = ids.len();
    let n_embd = *wte_weight_shape.get(1).unwrap();  // get(1) gets the number of COLUMNS since tenferro's COLmajor
    assert_eq!(n_embd, 768);
    let mut x_raw = Vec::new();
    for i in 0..n_embd {  // COLmajor!
        for id in &ids {
            x_raw.push(*wte_weight.get(&[*id, i])?);  // get([ROW,COLUMN])
        }
    }
    let x = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], x_raw).unwrap();


    Ok(())
}
