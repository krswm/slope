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
    println!("                        {t11:16.6e}{t12:16.6e}        ........{t18:16.6e}{t19:16.6e} ^");

    let t21 = tensor.get(&[1, 0]).unwrap();
    let t22 = tensor.get(&[1, 1]).unwrap();
    let t28 = tensor.get(&[1, shape1 - 2]).unwrap();
    let t29 = tensor.get(&[1, shape1 - 1]).unwrap();
    println!("                        {t21:16.6e}{t22:16.6e}        ........{t28:16.6e}{t29:16.6e} |");

    println!("{label:>21} =         ........        ........        ........        ........        ........ {shape0}");

    let t81 = tensor.get(&[shape0 - 2, 0]).unwrap();
    let t82 = tensor.get(&[shape0 - 2, 1]).unwrap();
    let t88 = tensor.get(&[shape0 - 2, shape1 - 2]).unwrap();
    let t89 = tensor.get(&[shape0 - 2, shape1 - 1]).unwrap();
    println!("                        {t81:16.6e}{t82:16.6e}        ........{t88:16.6e}{t89:16.6e} |");

    let t91 = tensor.get(&[shape0 - 1, 0]).unwrap();
    let t92 = tensor.get(&[shape0 - 1, 1]).unwrap();
    let t98 = tensor.get(&[shape0 - 1, shape1 - 2]).unwrap();
    let t99 = tensor.get(&[shape0 - 1, shape1 - 1]).unwrap();
    println!("                        {t91:16.6e}{t92:16.6e}        ........{t98:16.6e}{t99:16.6e} v");

    println!("                        <{} {shape1:14} {}>", "-".repeat(31), "-".repeat(31));

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

    /*
    let n_embd_tensor = TypedTensor::<f32>::from_vec_col_major(vec![1], vec![n_embd as f32]).unwrap();
    let xb_mean = xb_reduce_sum.div(&n_embd_tensor, &mut backend).unwrap();
    */
    let mut xb_mean = xb_reduce_sum.clone();
    for value in xb_mean.iter_mut().unwrap() {
        *value /= n_embd as f32;
    }
    let xb_mean_brd = xb_mean.broadcast_in_dim(&[n_ids, n_embd], &[0], &mut backend).unwrap();
    show("xb_mean_brd", &xb_mean_brd);

    //           Sum((x_i - <x>)^2)     Sum(<x^2> - <x>^2)
    // Var(x) = -------------------- = --------------------
    //                   N                      N  
    //
    // I'll use the first formula
    //
    // N - 1 or N on the denominator?
    // N - 1's result matches with my Python implementation so I guess the correct one is N - 1
    // but where is this stated?
    //
    // No, My python implementation was wrong.
    // Though I don't see the change of the final result.
    // it may be just a subtle point that won't affect the result most of time
    // The original paper for the layernorm uses 1/N.
    // https://arxiv.org/pdf/1607.06450

    /*
    // https://tensor4all.org/tenferro-rs/guides/tensor-operations.html#map-iteration-and-parallelism
    let xb_fluctuation = xb.clone();
    for value in xb_fluctuation.iter_mut().unwrap() {
        *value *= (*value - xb_mean) * (*value - xb_mean);
    }
    println!("{xb_mean:?}");
    */

    /*
    // TODO: There may be a better way...
    let mut raw_xb_fluctuation = Vec::new();
    for col in 0..n_embd {
        for row in 0..n_ids {
            let mut a = *xb.get(&[row, col]).unwrap();
            a -= *xb_mean.get(&[row]).unwrap();
            a *= a;
            raw_xb_fluctuation.push(a);
        }
    }
    */

    // x - <x>
    let xb_diff = xb.sub(&xb_mean_brd, &mut backend).unwrap();
    show("xb_diff", &xb_diff);

    // (x - <x>)^2
    let xb_fluct = xb_diff.mul(&xb_diff, &mut backend).unwrap();
    show("xb_fluct", &xb_fluct);

    // Sum(x - <x>)^2
    let xb_fluct_sum = xb_fluct.reduce_sum(&[1], &mut backend).unwrap();
    println!("{xb_fluct_sum:?}");

    // Sum(x - <x>)^2 / N
    let mut xb_var = xb_fluct_sum.clone();
    for value in xb_var.iter_mut().unwrap() {
        *value /= n_embd as f32;
    }
    let xb_var_brd = xb_var.broadcast_in_dim(&[n_ids, n_embd], &[0], &mut backend).unwrap();
    show("xb_var_brd", &xb_var_brd);

    /*
    let xb_fluctuation = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], raw_xb_fluctuation).unwrap();

    let xb_fluctuation_reduced_sum = xb_fluctuation.reduce_sum(&[1], &mut backend).unwrap();

    let xb_var = xb_fluctuation_reduced_sum.div(&n_embd_tensor, &mut backend).unwrap();
    println!("{xb_var:?}");
    */

    // I need
    //
    //     x - Mean[x]
    // ---------------------
    //  √(Var[x] - epsilon)

    // Numerator is same as x_diff

    // Var[x] - epsilon
    const LINENORM_EPSILON: f32 = 1e-5;
    let mut xb_purt = xb_var_brd.clone();
    for value in xb_purt.iter_mut().unwrap() {
        *value += LINENORM_EPSILON;
    }
    show("xb_purt", &xb_purt);

    // √(Var[x] - epsilon)
    let xb_denomi = xb_purt.sqrt(&mut backend).unwrap();
    show("xb_denomi", &xb_denomi);

    //     x - Mean[x]
    // ---------------------
    //  √(Var[x] - epsilon)

    let xb_division = xb_diff.div(&xb_denomi, &mut backend).unwrap();
    show("xb_division", &xb_division);

    // LayerNorm[x] = xb_division * gamma + beta

    let ln_1_weight = tensors.get("h.0.ln_1.weight").unwrap();  // gamma
    let ln_1_bias = tensors.get("h.0.ln_1.bias").unwrap();  // beta

    let xb_division_mul_gamma = xb_division.mul(&ln_1_weight, &mut backend).unwrap();
    show("xb_division_mul_gamma", &xb_division_mul_gamma);

    let xc = xb_division_mul_gamma.add(&ln_1_bias, &mut backend).unwrap();
    show("xc", &xc);

    let attn_c_attn_weight = tensors.get("h.0.attn.c_attn.weight").unwrap();
    let attn_c_attn_bias = tensors.get("h.0.attn.c_attn.bias").unwrap();

    let xd = xc.matmul(&attn_c_attn_weight, &mut backend).unwrap();
    show("xd", &xd);

    let xe = xd.add(&attn_c_attn_bias, &mut backend).unwrap();
    show("xe", &xe);

    // I need tensor.split
    // tenferro doc says "currently missing"
    // https://tensor4all.org/tenferro-rs/spec/operation-categories.html

    // No-split workaround

    // TODO: Don't hardcode them. Read from config instead.
    let headsize = 64;  // "N"
    let n_head = 12;

    let mut raw_stacked: Vec<f32> = Vec::new();
    for i_head in 0..n_head {
        let mut raw_q = Vec::new();
        let mut raw_k = Vec::new();
        let mut raw_v = Vec::new();

        // think, it's column major, what order do i have to put them?
        for a in 0..headsize {
            for row in 0..n_ids {
                raw_q.push(*xe.get(&[row, 0 * n_embd + headsize * i_head + a]).unwrap());
                raw_k.push(*xe.get(&[row, 1 * n_embd + headsize * i_head + a]).unwrap());
                raw_v.push(*xe.get(&[row, 2 * n_embd + headsize * i_head + a]).unwrap());
            }
        }

        let q = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, headsize], raw_q).unwrap();
        let k = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, headsize], raw_k).unwrap();
        let v = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, headsize], raw_v).unwrap();

        if i_head == 0 {
            show("q", &q);
            show("k", &k);
            show("v", &v);
        }

        let kt = k.transpose(&[1, 0], &mut backend).unwrap();
        if i_head == 0 {
            show("kt", &kt);
        }

        let mut xi = q.matmul(&kt, &mut backend).unwrap();
        for value in xi.iter_mut().unwrap() {
            *value /= (headsize as f32).sqrt();
        }
        if i_head == 0 {
            show("xi", &xi);
        }

        // Build a triangular matrix
        //
        //     0 -1e12 -1e12 ..... -1e12
        //     0     0 -1e12 ..... -1e12
        //     0     0     0 ..... -1e12
        // ..... ..... ..... ..... .....
        //     0     0     0 .....     0
        //
        // <causal mask matrix>

        let mut raw_causal_mask = Vec::new();
        for col in 0..n_ids {
            for row in 0..n_ids {
                raw_causal_mask.push(
                    if col <= row {
                        0.0
                    } else {
                        -1.0e12  // almost -inf
                    }
                )
            }
        }
        let causal_mask = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_ids], raw_causal_mask).unwrap();
        if i_head == 0 {
            show("causal_mask", &causal_mask);
        }

        let xii = xi.add(&causal_mask, &mut backend).unwrap();
        if i_head == 0 {
            show("xii", &xii);
        }

        /*
        let max_xii = xii.reduce_max(&[0, 1]).unwrap();
        if i_head == 0 {
            println!("{max_xii}");
        }

        // reduce_max not available for TypedTensor
        // but for TracedTensor
        */
        /*
        let max_xii = xii.iter()?.fold(-1.0e12f32, |a, b| a.max(*b));
        if i_head == 0 {
            println!("{max_xii}");
        }
        */

        let mut raw_max_xii = Vec::new();
        for row in 0..n_ids {
            let mut max = -1.0e12f32;
            for col in 0..n_ids{
                max = max.max(*xii.get(&[row, col])?);
            }
            raw_max_xii.push(max);
        }
        let max_xii = TypedTensor::<f32>::from_vec_col_major(vec![n_ids], raw_max_xii)?.broadcast_in_dim(&[n_ids, n_ids], &[0], &mut backend)?;
        if i_head == 0 {
            show("max_xii", &max_xii);
        }

        let negative_shift_xii = xii.sub(&max_xii, &mut backend)?;

        if i_head == 0 {
            show("negative_shift_xii", &negative_shift_xii);
        }

        let e = negative_shift_xii.exp(&mut backend)?;
        if i_head == 0 {
            show("e", &e);
        }

        let e_sum = e.reduce_sum(&[1], &mut backend)?.broadcast_in_dim(&[n_ids, n_ids], &[0], &mut backend)?;
        if i_head == 0 {
            show("e_sum", &e_sum);
        }

        let xj = e.div(&e_sum, &mut backend)?;
        if i_head == 0 {
            show("xj", &xj);
        }

        let xk = xj.matmul(&v, &mut backend)?;
        if i_head == 0 || i_head == 1 {
            show("xk", &xk);
        }

        // I need something similar to hstack in Pytorch
        // hstack = horizontal stack
        // I can utilize the column majorness of tenferro!
        raw_stacked.extend(xk.as_slice()?);
    }
    let stacked = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], raw_stacked)?;
    show("stacked", &stacked);

    Ok(())
}
