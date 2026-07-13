use std::collections::HashMap;
use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub fn transform(
    tensors: &HashMap<String, TypedTensor<f32>>,
    wte_weight_transposed: &TypedTensor<f32>,
    n_ctx: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    vocab_size: usize,
    ids: &Vec<usize>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let wte_weight = &tensors["wte.weight"];
    if wte_weight.shape() != &[vocab_size, n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    // wte_weight[ids]
    let x0 = {
        let mut colmaj = Vec::with_capacity(ids.len() * n_embd);
        for col in 0..n_embd {
            for row in ids {
                colmaj.push(*wte_weight.get(&[*row, col])?);
            }
        }
        TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], colmaj)?
    };

    let wpe_weight = &tensors["wpe.weight"];
    if wpe_weight.shape() != &[n_ctx, n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    // wpe_weight[range(len(ids))]
    let x1 = {
        let mut colmaj = Vec::with_capacity(ids.len() * n_embd);
        for col in 0..n_embd {
            for row in 0..ids.len() {
                colmaj.push(*wpe_weight.get(&[row, col])?);
            }
        }
        TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], colmaj)?
    };

    // wte[ids] + wpe[range(len(ids))]
    let mut x2 = x0.add(&x1, backend)?;

    for i_layer in 0..n_layer {
        let ln_1_weight = &tensors[&format!("h.{i_layer}.ln_1.weight")];
        if ln_1_weight.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let ln_1_bias = &tensors[&format!("h.{i_layer}.ln_1.bias")];
        if ln_1_bias.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let x3 = layer_norm(
            &x2,
            ln_1_weight,
            ln_1_bias,
            n_embd,
            ids.len(),
            backend,
            i_layer == 0,
        );

        let attn_c_attn_weight = &tensors[&format!("h.{i_layer}.attn.c_attn.weight")];
        if attn_c_attn_weight.shape() != &[n_embd, 3 * n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x3 @ attn_c_attn_weight
        let x4 = x3.matmul(&attn_c_attn_weight, backend)?;

        let attn_c_attn_bias = &tensors[&format!("h.{i_layer}.attn.c_attn.bias")];
        if attn_c_attn_bias.shape() != &[3 * n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x3 @ attn_c_attn_weight + attn_c_attn_bias
        let x5 = x4.add(&attn_c_attn_bias, backend)?;

        let x6 = {
            let size_of_head = n_embd / n_head;
            let mut raw_stacked: Vec<f32> = Vec::with_capacity(ids.len() * n_embd);
            for i_head in 0..n_head {
                let (q, k, v) = {
                    let mut q_colmaj = Vec::with_capacity(ids.len() * size_of_head);
                    let mut k_colmaj = Vec::with_capacity(ids.len() * size_of_head);
                    let mut v_colmaj = Vec::with_capacity(ids.len() * size_of_head);

                    for col in 0..size_of_head {
                        for row in 0..ids.len() {
                            q_colmaj
                                .push(*x5.get(&[row, 0 * n_embd + size_of_head * i_head + col])?);
                            k_colmaj
                                .push(*x5.get(&[row, 1 * n_embd + size_of_head * i_head + col])?);
                            v_colmaj
                                .push(*x5.get(&[row, 2 * n_embd + size_of_head * i_head + col])?);
                        }
                    }

                    (
                        TypedTensor::<f32>::from_vec_col_major(
                            vec![ids.len(), size_of_head],
                            q_colmaj,
                        )?,
                        TypedTensor::<f32>::from_vec_col_major(
                            vec![ids.len(), size_of_head],
                            k_colmaj,
                        )?,
                        TypedTensor::<f32>::from_vec_col_major(
                            vec![ids.len(), size_of_head],
                            v_colmaj,
                        )?,
                    )
                };

                // kᵀ
                let x7 = k.transpose(&[1, 0], backend)?;

                // q @ kᵀ
                let x8 = q.matmul(&x7, backend).unwrap();

                // √size_of_head
                let x9 = TypedTensor::<f32>::from_vec_col_major(
                    vec![1, 1],
                    vec![(size_of_head as f32).sqrt()],
                )?;

                // q @ kᵀ / √size_of_head
                let x10 = x8.div(&x9, backend)?;

                //           x10                       x11
                // ⎛ a₁₁ a₁₂ a₁₃ ……… a₁ₙ ⎞   ⎛ a₁₁  −∞  −∞ ………  −∞ ⎞
                // ⎜ a₂₁ a₂₂ a₂₃ ……… a₂ₙ ⎟   ⎜ a₂₁ a₂₂  −∞ ………  −∞ ⎟
                // ⎜ a₃₁ a₃₂ a₃₃ ……… a₃ₙ ⎟ → ⎜ a₃₁ a₃₂ a₃₃ ………  −∞ ⎟
                // ⎜ ……… ……… ……… ……… ……… ⎟   ⎜ ……… ……… ……… ……… ……… ⎟
                // ⎝ aₙ₁ aₙ₂ aₙ₃ ……… aₙₙ ⎠   ⎝ aₙ₁ aₙ₂ aₙ₃ ……… aₙₙ ⎠
                let x11 = {
                    let mut colmaj = Vec::with_capacity(ids.len() * ids.len());
                    for col in 0..ids.len() {
                        for row in 0..ids.len() {
                            colmaj.push(if col <= row {
                                *x10.get(&[row, col])?
                            } else {
                                f32::NEG_INFINITY
                            })
                        }
                    }
                    TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), ids.len()], colmaj)?
                };

                // max(x11)
                let x12 = {
                    let mut colmaj = Vec::with_capacity(ids.len());
                    for row in 0..ids.len() {
                        let mut rowvec = Vec::with_capacity(ids.len());
                        for col in 0..ids.len() {
                            rowvec.push(*x11.get(&[row, col])?);
                        }
                        let rowmax = rowvec
                            .iter()
                            .copied()
                            .max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap();
                        colmaj.push(rowmax);
                    }
                    TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), 1], colmaj)?
                };

                // x11 - max(x11)
                let x13 = x11.sub(&x12, backend)?;

                // exp(x11 - max(x11))
                let x14 = x13.exp(backend)?;

                // sum(exp(x11 - max(x11)))
                let x15 = x14
                    .reduce_sum(&[1], backend)?
                    .reshape(&[ids.len(), 1], backend)?;

                // exp(x11 - max(x11)) / sum(exp(x11 - max(x11))) = softmax(x11)
                let x16 = x14.div(&x15, backend)?;

                // softmax(x11) @ v
                let x17 = x16.matmul(&v, backend)?;

                raw_stacked.extend(x17.as_slice()?);
            }
            TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], raw_stacked)?
        };

        let attn_c_proj_weight = tensors
            .get(&format!("h.{i_layer}.attn.c_proj.weight"))
            .unwrap();
        let attn_c_proj_bias = tensors
            .get(&format!("h.{i_layer}.attn.c_proj.bias"))
            .unwrap();

        let xl = x6.matmul(&attn_c_proj_weight, backend).unwrap();

        let xm = xl.add(&attn_c_proj_bias, backend).unwrap();

        let xn = x2.add(&xm, backend).unwrap();

        let ln_2_weight = tensors.get(&format!("h.{i_layer}.ln_2.weight")).unwrap(); // gamma
        let ln_2_bias = tensors.get(&format!("h.{i_layer}.ln_2.bias")).unwrap(); // beta
        let xo = layer_norm(
            &xn,
            ln_2_weight,
            ln_2_bias,
            n_embd,
            ids.len(),
            backend,
            false,
        );

        let mlp_c_fc_weight = tensors
            .get(&format!("h.{i_layer}.mlp.c_fc.weight"))
            .unwrap();
        let mlp_c_fc_bias = tensors.get(&format!("h.{i_layer}.mlp.c_fc.bias")).unwrap();

        let xp = xo.matmul(&mlp_c_fc_weight, backend).unwrap();

        let xq = xp.add(&mlp_c_fc_bias, backend).unwrap();

        // GELU
        // The approximation coefficients are according to the original paper that introduced Gelu:
        // https://arxiv.org/pdf/1606.08415
        let xq_squared = xq.mul(&xq, backend).unwrap();
        let xq_cubed = xq_squared.mul(&xq, backend).unwrap();
        let xq_cubed_coef_as_tensor =
            TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![0.044715])?;
        let xq_cubed_scaled = xq_cubed.mul(&xq_cubed_coef_as_tensor, backend)?;
        let inside_tanh = xq.add(&xq_cubed_scaled, backend).unwrap();
        let inside_tanh_coef_as_tensor = TypedTensor::<f32>::from_vec_col_major(
            vec![1, 1],
            vec![(2.0f32 / 3.1415926535897932385f32).sqrt()],
        )?;
        let inside_tanh_scaled = inside_tanh.mul(&inside_tanh_coef_as_tensor, backend)?;
        let tanhed = inside_tanh_scaled.tanh(backend).unwrap();
        let one_as_tensor = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![1.0])?;
        let tanhed_raised = tanhed.add(&one_as_tensor, backend)?;
        let xr = xq.mul(&tanhed_raised, backend).unwrap();
        let xr_coef_as_tensor = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![0.5])?;
        let xr_scaled = xr.mul(&xr_coef_as_tensor, backend)?;

        let mlp_c_proj_weight = tensors
            .get(&format!("h.{i_layer}.mlp.c_proj.weight"))
            .unwrap();
        let mlp_c_proj_bias = tensors
            .get(&format!("h.{i_layer}.mlp.c_proj.bias"))
            .unwrap();

        let xs = xr_scaled.matmul(&mlp_c_proj_weight, backend).unwrap();

        let xt = xs.add(&mlp_c_proj_bias, backend).unwrap();

        x2 = xn.add(&xt, backend).unwrap();
    }

    let ln_f_weight = tensors.get("ln_f.weight").unwrap(); // gamma
    let ln_f_bias = tensors.get("ln_f.bias").unwrap(); // beta
    let xu = layer_norm(
        &x2,
        ln_f_weight,
        ln_f_bias,
        n_embd,
        ids.len(),
        backend,
        false,
    );

    // This is slow! That's undersandable: it's transpose of almost 50000x4 tensor it may not be fast
    // Solution is to precalculate transposed one.
    let xv = xu.matmul(wte_weight_transposed, backend).unwrap();

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
