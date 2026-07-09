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

use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub mod safetensors_to_tenferro;

fn show(label: &str, tensor: &TypedTensor<f32>) {
    // Show a tensor for debug.

    let shape0 = tensor.shape().get(0).unwrap();
    let shape1 = tensor.shape().get(1).unwrap();

    // What have I done! Rust is 0 based!!!

    let t11 = tensor.get(&[0, 0]).unwrap();
    let t12 = tensor.get(&[0, 1]).unwrap();
    let t18 = tensor.get(&[0, shape1 - 2]).unwrap();
    let t19 = tensor.get(&[0, shape1 - 1]).unwrap();
    println!("                {t11:16.6}{t12:16.6}        ........{t18:16.6}{t19:16.6} ^");

    let t21 = tensor.get(&[1, 0]).unwrap();
    let t22 = tensor.get(&[1, 1]).unwrap();
    let t28 = tensor.get(&[1, shape1 - 2]).unwrap();
    let t29 = tensor.get(&[1, shape1 - 1]).unwrap();
    println!("                {t21:16.6}{t22:16.6}        ........{t28:16.6}{t29:16.6} |");

    println!("{label:>13} =         ........        ........        ........        ........        ........ {shape0}");

    let t81 = tensor.get(&[shape0 - 2, 0]).unwrap();
    let t82 = tensor.get(&[shape0 - 2, 1]).unwrap();
    let t88 = tensor.get(&[shape0 - 2, shape1 - 2]).unwrap();
    let t89 = tensor.get(&[shape0 - 2, shape1 - 1]).unwrap();
    println!("                {t81:16.6}{t82:16.6}        ........{t88:16.6}{t89:16.6} |");

    let t91 = tensor.get(&[shape0 - 1, 0]).unwrap();
    let t92 = tensor.get(&[shape0 - 1, 1]).unwrap();
    let t98 = tensor.get(&[shape0 - 1, shape1 - 2]).unwrap();
    let t99 = tensor.get(&[shape0 - 1, shape1 - 1]).unwrap();
    println!("                {t91:16.6}{t92:16.6}        ........{t98:16.6}{t99:16.6} v");

    println!("                <{} {shape1:14} {}>", "-".repeat(31), "-".repeat(31));

    println!();
}

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
    let xa = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], x_raw).unwrap();
    // show("xa", &xa);

    let wpe_weight = tensors.get("wpe.weight").unwrap();
    let n_ctx = *wpe_weight.shape().get(0).unwrap();
    assert_eq!(n_ctx, 1024);
    assert!(n_ids < n_ctx);

    let mut raw_sliced_wpe_weight = Vec::new();
    for col in 0..n_embd {
        for row in 0..n_ids {
            raw_sliced_wpe_weight.push(*wpe_weight.get(&[row, col])?);
        }
    }
    let sliced_wpe_weight = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], raw_sliced_wpe_weight).unwrap();

    let mut backend = tenferro_cpu::CpuBackend::new();

    let xb = xa.add(&sliced_wpe_weight, &mut backend).unwrap();
    show("xb", &xb);

    // Do we have mean and var in tenferro?
    //
    // The tenferro doc says "audit pending" for mean?
    // https://tensor4all.org/tenferro-rs/spec/operation-categories.html#reductions 
    //
    // I couldn't find mean and var in tenferro rust doc
    // https://tensor4all.org/tenferro-rs/api/tenferro_runtime/index.html

    // Do it manually. Let's go! 
    
    /*
    let mut raw_xb_rowwise_sum = Vec::new();
    for row in 0..n_ids {
        let mut sum = 0.0;
        for col in 0..n_embd {
            sum += *xb.get(&[row, col])?;
        }
        raw_xb_rowwise_sum.push(sum);
    }
    let xb_rowwise_sum = TypedTensor::<f32>::from_vec_col_major(vec![n_ids], raw_xb_rowwise_sum).unwrap();
    println!("{xb_rowwise_sum:?}");
    */

    let xb_reduce_sum = xb.reduce_sum(&[1], &mut backend).unwrap();
    println!("{xb_reduce_sum:?}");

    let n_embd_tensor = TypedTensor::<f32>::from_vec_col_major(vec![1], vec![n_embd as f32]).unwrap();
    let xb_mean = xb_reduce_sum.div(&n_embd_tensor, &mut backend).unwrap();
    println!("{xb_mean:?}");

    Ok(())
}
