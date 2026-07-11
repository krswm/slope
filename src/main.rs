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

use std::io::Write;
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
    let args: Vec<String> = std::env::args().collect();

    let safetensors_path = &format!("{}/model.safetensors", &args[1]);
    let vocab_path = &format!("{}/vocab.json", &args[1]);

    let vocab_raw_json = std::fs::read_to_string(vocab_path)?;
    let vocab = json::parse(&vocab_raw_json)?;

    let mut id_to_token = std::collections::HashMap::<usize, &str>::new();
    for (token, id) in vocab.entries() {
        id_to_token.insert(id.as_usize().unwrap(), token);
    }

    let tensors = safetensors_to_tenferro::st_to_tf::st_to_tf(safetensors_path)?;

    // looks messy...
    let mut ids = args[2..].into_iter().map(|id| id.parse::<usize>().unwrap()).collect::<Vec<usize>>();

    let n_vocab = 50257;  // TODO: Don't hardcode this!

    for id in &ids {
        print!("\x1b[1m{}\x1b[22m", replace_characters(id_to_token.get(&id).unwrap()));
        std::io::stdout().flush().unwrap();
    }


    for _ in 0..1000 {

        let a = gpt(&tensors, &ids)?;

        let mut next_id = 0;
        let mut max = -1.0e12f32;  // I'll do greedy sampling
        for col in 0..n_vocab {
            let b = *a.get(&[ids.len() - 1, col])?;
            if b > max {
                max = b;
                next_id = col;
            }
        }

        print!("\x1b[1;35m{}\x1b[22;39m", replace_characters(id_to_token.get(&next_id).unwrap()));
        std::io::stdout().flush().unwrap();

        ids.push(next_id);
    }
    println!();

    // hooray! It works!
    // Very slow anyways
    // Needs refactoring!

    Ok(())
}

fn replace_characters(text: &str) -> String {
    text.chars().map( |ch|
        if ch as u32 >= 0x100 { char::from_u32(ch as u32 - 0x100).unwrap() } else { ch }
    ).collect()
}

fn gpt(tensors: &std::collections::HashMap<String, TypedTensor<f32>>, ids: &Vec<usize>) -> Result<TypedTensor<f32>, Box<dyn std::error::Error>> {
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
        for id in ids {
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

    let mut xb = xa.add(&sliced_wpe_weight, &mut backend).unwrap();

    let n_layer = 12;  // TODO: Do not hardcode it!
    for i in 0..n_layer {
        let ln_1_weight = tensors.get(&format!("h.{i}.ln_1.weight")).unwrap();  // gamma
        let ln_1_bias = tensors.get(&format!("h.{i}.ln_1.bias")).unwrap();  // beta
        let xc = layer_norm(&xb, ln_1_weight, ln_1_bias, n_embd, n_ids, &mut backend);

        let attn_c_attn_weight = tensors.get(&format!("h.{i}.attn.c_attn.weight")).unwrap();
        let attn_c_attn_bias = tensors.get(&format!("h.{i}.attn.c_attn.bias")).unwrap();

        let xd = xc.matmul(&attn_c_attn_weight, &mut backend).unwrap();

        let xe = xd.add(&attn_c_attn_bias, &mut backend).unwrap();

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

            let kt = k.transpose(&[1, 0], &mut backend).unwrap();

            let mut xi = q.matmul(&kt, &mut backend).unwrap();
            for value in xi.iter_mut().unwrap() {
                *value /= (headsize as f32).sqrt();
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

            let xii = xi.add(&causal_mask, &mut backend).unwrap();

            let mut raw_max_xii = Vec::new();
            for row in 0..n_ids {
                let mut max = -1.0e12f32;
                for col in 0..n_ids{
                    max = max.max(*xii.get(&[row, col])?);
                }
                raw_max_xii.push(max);
            }
            let max_xii = TypedTensor::<f32>::from_vec_col_major(vec![n_ids], raw_max_xii)?.broadcast_in_dim(&[n_ids, n_ids], &[0], &mut backend)?;

            let negative_shift_xii = xii.sub(&max_xii, &mut backend)?;

            let e = negative_shift_xii.exp(&mut backend)?;

            let e_sum = e.reduce_sum(&[1], &mut backend)?.broadcast_in_dim(&[n_ids, n_ids], &[0], &mut backend)?;

            let xj = e.div(&e_sum, &mut backend)?;

            let xk = xj.matmul(&v, &mut backend)?;

            raw_stacked.extend(xk.as_slice()?);
        }
        let stacked = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], raw_stacked)?;

        let attn_c_proj_weight = tensors.get(&format!("h.{i}.attn.c_proj.weight")).unwrap();
        let attn_c_proj_bias = tensors.get(&format!("h.{i}.attn.c_proj.bias")).unwrap();

        let xl = stacked.matmul(&attn_c_proj_weight, &mut backend).unwrap();

        let xm = xl.add(&attn_c_proj_bias, &mut backend).unwrap();

        let xn = xb.add(&xm, &mut backend).unwrap();

        let ln_2_weight = tensors.get(&format!("h.{i}.ln_2.weight")).unwrap();  // gamma
        let ln_2_bias = tensors.get(&format!("h.{i}.ln_2.bias")).unwrap();  // beta
        let xo = layer_norm(&xn, ln_2_weight, ln_2_bias, n_embd, n_ids, &mut backend);

        let mlp_c_fc_weight = tensors.get(&format!("h.{i}.mlp.c_fc.weight")).unwrap();
        let mlp_c_fc_bias = tensors.get(&format!("h.{i}.mlp.c_fc.bias")).unwrap();

        let xp = xo.matmul(&mlp_c_fc_weight, &mut backend).unwrap();

        let xq = xp.add(&mlp_c_fc_bias, &mut backend).unwrap();

        // GELU
        // The approximation coefficients are according to the original paper that introduced Gelu:
        // https://arxiv.org/pdf/1606.08415
        let xq_squared = xq.mul(&xq, &mut backend).unwrap();
        let mut xq_cubed = xq_squared.mul(&xq, &mut backend).unwrap();
        for value in xq_cubed.iter_mut().unwrap() {
            *value *= 0.044715;
        }
        let mut inside_tanh = xq.add(&xq_cubed, &mut backend).unwrap();
        for value in inside_tanh.iter_mut().unwrap() {
            *value *= (2.0f32 / 3.1415926535897932385f32).sqrt()
        }
        let mut tanhed = inside_tanh.tanh(&mut backend).unwrap();
        for value in tanhed.iter_mut().unwrap() {
            *value += 1.0f32;
        }
        let mut xr = xq.mul(&tanhed, &mut backend).unwrap();
        for value in xr.iter_mut().unwrap() {
            *value *= 0.5;
        }

        let mlp_c_proj_weight = tensors.get(&format!("h.{i}.mlp.c_proj.weight")).unwrap();
        let mlp_c_proj_bias = tensors.get(&format!("h.{i}.mlp.c_proj.bias")).unwrap();

        let xs = xr.matmul(&mlp_c_proj_weight, &mut backend).unwrap();

        let xt = xs.add(&mlp_c_proj_bias, &mut backend).unwrap();

        xb = xn.add(&xt, &mut backend).unwrap();

        // FeedForward done!
    }

    let ln_f_weight = tensors.get("ln_f.weight").unwrap();  // gamma
    let ln_f_bias = tensors.get("ln_f.bias").unwrap();  // beta
    let xu = layer_norm(&xb, ln_f_weight, ln_f_bias, n_embd, n_ids, &mut backend);

    let wte_weight_transposed = wte_weight.transpose(&[1, 0], &mut backend).unwrap();
    let xv = xu.matmul(&wte_weight_transposed, &mut backend).unwrap();

    // Yaaay!

    Ok(xv)
}


fn layer_norm(xb: &TypedTensor<f32>, weight: &TypedTensor<f32>, bias: &TypedTensor<f32>, n_embd: usize, n_ids: usize, backend: &mut tenferro_cpu::CpuBackend) -> TypedTensor<f32> {
    let xb_reduce_sum = xb.reduce_sum(&[1], backend).unwrap();

    let mut xb_mean = xb_reduce_sum.clone();
    for value in xb_mean.iter_mut().unwrap() {
        *value /= n_embd as f32;
    }
    let xb_mean_brd = xb_mean.broadcast_in_dim(&[n_ids, n_embd], &[0], backend).unwrap();

    //           Sum((x_i - <x>)^2)     Sum(<x^2> - <x>^2)
    // Var(x) = -------------------- = --------------------
    //                   N                      N  
    //
    // I'll use the first formula
    // The original paper for the layernorm uses 1/N.
    // https://arxiv.org/pdf/1607.06450

    // x - <x>
    let xb_diff = xb.sub(&xb_mean_brd, backend).unwrap();

    // (x - <x>)^2
    let xb_fluct = xb_diff.mul(&xb_diff, backend).unwrap();

    // Sum(x - <x>)^2
    let xb_fluct_sum = xb_fluct.reduce_sum(&[1], backend).unwrap();

    // Sum(x - <x>)^2 / N
    let mut xb_var = xb_fluct_sum.clone();
    for value in xb_var.iter_mut().unwrap() {
        *value /= n_embd as f32;
    }
    let xb_var_brd = xb_var.broadcast_in_dim(&[n_ids, n_embd], &[0], backend).unwrap();

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

    // √(Var[x] - epsilon)
    let xb_denomi = xb_purt.sqrt(backend).unwrap();

    //     x - Mean[x]
    // ---------------------
    //  √(Var[x] - epsilon)

    let xb_division = xb_diff.div(&xb_denomi, backend).unwrap();

    // LayerNorm[x] = xb_division * gamma + beta

    let xb_division_mul_gamma = xb_division.mul(&weight, backend).unwrap();

    let xc = xb_division_mul_gamma.add(&bias, backend).unwrap();

    xc
}
