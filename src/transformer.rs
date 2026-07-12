use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub fn transform(
    tensors: &std::collections::HashMap<String, TypedTensor<f32>>,
    n_ctx: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    vocab_size: usize,
    ids: &Vec<usize>,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    // (a) Embedding

    let wte_weight = &tensors["wte.weight"];
    assert_eq!(wte_weight.shape(), &[vocab_size, n_embd]);

    let a0 = {
        let mut colmaj = Vec::with_capacity(ids.len() * n_embd);
        for i in 0..n_embd {
            for id in ids {
                colmaj.push(*wte_weight.get(&[*id, i])?);
            }
        }
        TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], colmaj)?
    };

    let wpe_weight = &tensors["wpe.weight"];
    assert_eq!(wpe_weight.shape(), &[n_ctx, n_embd]);

    let a1 = {
        let mut colmaj = Vec::with_capacity(ids.len() * n_embd);
        for col in 0..n_embd {
            for row in 0..ids.len() {
                colmaj.push(*wpe_weight.get(&[row, col])?);
            }
        }
        TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], colmaj)?
    };

    let mut a2 = a0.add(&a1, &mut backend)?;

    let n_ids = ids.len();

    for i in 0..n_layer {
        // (b) Multi-Head Attention

        let ln_1_weight = &tensors[&format!("h.{i}.ln_1.weight")];
        assert_eq!(ln_1_weight.shape(), &[n_embd]);

        let ln_1_bias = &tensors[&format!("h.{i}.ln_1.bias")];
        assert_eq!(ln_1_bias.shape(), &[n_embd]);

        let b0 = layer_norm(
            &a2,
            ln_1_weight,
            ln_1_bias,
            n_embd,
            n_ids,
            &mut backend,
            i == 0,
        );

        let attn_c_attn_weight = &tensors[&format!("h.{i}.attn.c_attn.weight")];
        assert_eq!(attn_c_attn_weight.shape(), &[n_embd, 3 * n_embd]);

        let b1 = b0.matmul(&attn_c_attn_weight, &mut backend)?;

        let attn_c_attn_bias = &tensors[&format!("h.{i}.attn.c_attn.bias")];
        assert_eq!(attn_c_attn_bias.shape(), &[3 * n_embd]);

        let b2 = b1.add(&attn_c_attn_bias, &mut backend)?;

        // I need tensor.split
        // tenferro doc says "currently missing"
        // https://tensor4all.org/tenferro-rs/spec/operation-categories.html

        // No-split workaround

        let headsize = n_embd / n_head; // "N"

        let mut raw_stacked: Vec<f32> = Vec::new();
        for i_head in 0..n_head {
            let mut raw_q = Vec::new();
            let mut raw_k = Vec::new();
            let mut raw_v = Vec::new();

            for a in 0..headsize {
                for row in 0..n_ids {
                    raw_q.push(*b2.get(&[row, 0 * n_embd + headsize * i_head + a]).unwrap());
                    raw_k.push(*b2.get(&[row, 1 * n_embd + headsize * i_head + a]).unwrap());
                    raw_v.push(*b2.get(&[row, 2 * n_embd + headsize * i_head + a]).unwrap());
                }
            }

            let q = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, headsize], raw_q).unwrap();
            let k = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, headsize], raw_k).unwrap();
            let v = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, headsize], raw_v).unwrap();

            let kt = k.transpose(&[1, 0], &mut backend).unwrap();

            let headsize_sqrt_as_tensor =
                TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![(headsize as f32).sqrt()])?;

            let xi = q.matmul(&kt, &mut backend).unwrap();

            let xi_div = xi.div(&headsize_sqrt_as_tensor, &mut backend)?;

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
                    raw_causal_mask.push(if col <= row {
                        0.0
                    } else {
                        -1.0e12 // almost -inf
                    })
                }
            }
            let causal_mask =
                TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_ids], raw_causal_mask)
                    .unwrap();

            let xii = xi_div.add(&causal_mask, &mut backend).unwrap();

            let mut raw_max_xii = Vec::new();
            for row in 0..n_ids {
                let mut max = -1.0e12f32;
                for col in 0..n_ids {
                    max = max.max(*xii.get(&[row, col])?);
                }
                raw_max_xii.push(max);
            }
            let max_xii = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, 1], raw_max_xii)?;

            let negative_shift_xii = xii.sub(&max_xii, &mut backend)?;

            let e = negative_shift_xii.exp(&mut backend)?;

            let e_sum = e
                .reduce_sum(&[1], &mut backend)?
                .reshape(&[n_ids, 1], &mut backend)?;

            let xj = e.div(&e_sum, &mut backend)?;

            let xk = xj.matmul(&v, &mut backend)?;

            raw_stacked.extend(xk.as_slice()?);
        }
        let stacked = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], raw_stacked)?;

        let attn_c_proj_weight = tensors.get(&format!("h.{i}.attn.c_proj.weight")).unwrap();
        let attn_c_proj_bias = tensors.get(&format!("h.{i}.attn.c_proj.bias")).unwrap();

        let xl = stacked.matmul(&attn_c_proj_weight, &mut backend).unwrap();

        let xm = xl.add(&attn_c_proj_bias, &mut backend).unwrap();

        let xn = a2.add(&xm, &mut backend).unwrap();

        let ln_2_weight = tensors.get(&format!("h.{i}.ln_2.weight")).unwrap(); // gamma
        let ln_2_bias = tensors.get(&format!("h.{i}.ln_2.bias")).unwrap(); // beta
        let xo = layer_norm(
            &xn,
            ln_2_weight,
            ln_2_bias,
            n_embd,
            n_ids,
            &mut backend,
            false,
        );

        let mlp_c_fc_weight = tensors.get(&format!("h.{i}.mlp.c_fc.weight")).unwrap();
        let mlp_c_fc_bias = tensors.get(&format!("h.{i}.mlp.c_fc.bias")).unwrap();

        let xp = xo.matmul(&mlp_c_fc_weight, &mut backend).unwrap();

        let xq = xp.add(&mlp_c_fc_bias, &mut backend).unwrap();

        // GELU
        // The approximation coefficients are according to the original paper that introduced Gelu:
        // https://arxiv.org/pdf/1606.08415
        let xq_squared = xq.mul(&xq, &mut backend).unwrap();
        let xq_cubed = xq_squared.mul(&xq, &mut backend).unwrap();
        let xq_cubed_coef_as_tensor =
            TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![0.044715])?;
        let xq_cubed_scaled = xq_cubed.mul(&xq_cubed_coef_as_tensor, &mut backend)?;
        let inside_tanh = xq.add(&xq_cubed_scaled, &mut backend).unwrap();
        let inside_tanh_coef_as_tensor = TypedTensor::<f32>::from_vec_col_major(
            vec![1, 1],
            vec![(2.0f32 / 3.1415926535897932385f32).sqrt()],
        )?;
        let inside_tanh_scaled = inside_tanh.mul(&inside_tanh_coef_as_tensor, &mut backend)?;
        let tanhed = inside_tanh_scaled.tanh(&mut backend).unwrap();
        let one_as_tensor = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![1.0])?;
        let tanhed_raised = tanhed.add(&one_as_tensor, &mut backend)?;
        let xr = xq.mul(&tanhed_raised, &mut backend).unwrap();
        let xr_coef_as_tensor = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![0.5])?;
        let xr_scaled = xr.mul(&xr_coef_as_tensor, &mut backend)?;

        let mlp_c_proj_weight = tensors.get(&format!("h.{i}.mlp.c_proj.weight")).unwrap();
        let mlp_c_proj_bias = tensors.get(&format!("h.{i}.mlp.c_proj.bias")).unwrap();

        let xs = xr_scaled.matmul(&mlp_c_proj_weight, &mut backend).unwrap();

        let xt = xs.add(&mlp_c_proj_bias, &mut backend).unwrap();

        a2 = xn.add(&xt, &mut backend).unwrap();
    }

    let ln_f_weight = tensors.get("ln_f.weight").unwrap(); // gamma
    let ln_f_bias = tensors.get("ln_f.bias").unwrap(); // beta
    let xu = layer_norm(
        &a2,
        ln_f_weight,
        ln_f_bias,
        n_embd,
        n_ids,
        &mut backend,
        false,
    );

    let wte_weight_transposed = wte_weight.transpose(&[1, 0], &mut backend).unwrap();
    let xv = xu.matmul(&wte_weight_transposed, &mut backend).unwrap();

    Ok(xv)
}

fn layer_norm(
    xb: &TypedTensor<f32>,
    weight: &TypedTensor<f32>,
    bias: &TypedTensor<f32>,
    n_embd: usize,
    n_ids: usize,
    backend: &mut tenferro_cpu::CpuBackend,
    is_show: bool,
) -> TypedTensor<f32> {
    let xb_reduce_sum = xb.reduce_sum(&[1], backend).unwrap();

    let n_embd_as_tensor =
        TypedTensor::<f32>::from_vec_col_major(vec![1], vec![n_embd as f32]).unwrap();

    let xb_mean = xb_reduce_sum.div(&n_embd_as_tensor, backend).unwrap();

    let xb_mean_brd = xb_mean.reshape(&[n_ids, 1], backend).unwrap();

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
    let xb_var = xb_fluct_sum.div(&n_embd_as_tensor, backend).unwrap();

    // I need
    //
    //     x - Mean[x]
    // ---------------------
    //  √(Var[x] + epsilon)

    // Numerator is same as x_diff

    // Var[x] + epsilon
    const LINENORM_EPSILON: f32 = 1e-5;

    let linenorm_epsilon_as_tensor =
        TypedTensor::<f32>::from_vec_col_major(vec![1], vec![LINENORM_EPSILON]).unwrap();

    let xb_purt = xb_var.add(&linenorm_epsilon_as_tensor, backend).unwrap();

    // √(Var[x] - epsilon)
    let xb_denomi = xb_purt.sqrt(backend).unwrap();

    let xb_denomi_brd = xb_denomi.reshape(&[n_ids, 1], backend).unwrap();

    //     x - Mean[x]
    // ---------------------
    //  √(Var[x] - epsilon)

    let xb_division = xb_diff.div(&xb_denomi_brd, backend).unwrap();

    // LayerNorm[x] = xb_division * gamma + beta

    let xb_division_mul_gamma = xb_division.mul(&weight, backend).unwrap();

    let xc = xb_division_mul_gamma.add(&bias, backend).unwrap();

    xc
}

/// Pretty-print a 2D tensor for debug.
#[allow(dead_code)]
fn show(tensor: &TypedTensor<f32>) -> Result<(), Box<dyn Error>> {
    if tensor.shape().len() != 2 {
        return Err("not 2D tensor".into());
    }

    let num_rows = tensor.shape()[0];
    let num_cols = tensor.shape()[1];

    if num_rows == 0 {
        return Err("num_rows is 0".into());
    }
    if num_cols == 0 {
        return Err("num_cols is 0".into());
    }

    println!("┌{}┐", "─".repeat(39));

    println!(
        "│ {:15.6e} {:5} {:15.6e} │",
        tensor.get(&[0, 0]).unwrap(),
        "",
        tensor.get(&[0, num_cols - 1]).unwrap(),
    );

    println!("│ {:15} {:5} {:15} {num_rows}", "", "", "");

    println!(
        "│ {:15.6e} {:5} {:15.6e} │",
        tensor.get(&[num_rows - 1, 0]).unwrap(),
        "",
        tensor.get(&[num_rows - 1, num_cols - 1]).unwrap(),
    );

    println!("└{} {num_cols:5} {}┘", "─".repeat(16), "─".repeat(16));

    Ok(())
}
