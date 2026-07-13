use std::collections::HashMap;
use std::error::Error;
use std::f32::consts::PI;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

/// Transformer for the GPT-2 architecture
pub fn transform(
    tensors: &HashMap<String, TypedTensor<f32>>,
    transposed_wte_weight: &TypedTensor<f32>,
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

    // ==== Embedding ====

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
        // ==== Masked Multi-Head Attention ====

        let ln_1_weight = &tensors[&format!("h.{i_layer}.ln_1.weight")];
        if ln_1_weight.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let ln_1_bias = &tensors[&format!("h.{i_layer}.ln_1.bias")];
        if ln_1_bias.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let x3 = layer_norm(&x2, ln_1_weight, ln_1_bias, backend)?;

        let attn_c_attn_weight = &tensors[&format!("h.{i_layer}.attn.c_attn.weight")];
        if attn_c_attn_weight.shape() != &[n_embd, 3 * n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let attn_c_attn_bias = &tensors[&format!("h.{i_layer}.attn.c_attn.bias")];
        if attn_c_attn_bias.shape() != &[3 * n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x3 @ attn_c_attn_weight + attn_c_attn_bias
        let x4 = x3
            .matmul(&attn_c_attn_weight, backend)?
            .add(&attn_c_attn_bias, backend)?;

        let x15 = {
            let size_of_head = n_embd / n_head;
            let mut stacked_colmaj: Vec<f32> = Vec::with_capacity(ids.len() * n_embd);

            //      ├──── 0..n_head ────┼──── 0..n_head ────┼──── 0..n_head ────┤
            //      ┏━━━┯━━━┯   ┯━━━┯━━━┳━━━┯━━━┯   ┯━━━┯━━━┳━━━┯━━━┯   ┯━━━┯━━━┓ ┐
            // x4 = ┃ q │ q │ … │ q │ q ┃ k │ k │ … │ k │ k ┃ v │ v │ … │ v │ v ┃ ids.len() rows
            //      ┗━━━┷━━━┷   ┷━━━┷━━━┻━━━┷━━━┷   ┷━━━┷━━━┻━━━┷━━━┷   ┷━━━┷━━━┛ ┘
            //      └─┬─┘
            //        size_of_head columns

            for i_head in 0..n_head {
                let (q, k, v) = {
                    let mut q_colmaj = Vec::with_capacity(ids.len() * size_of_head);
                    let mut k_colmaj = Vec::with_capacity(ids.len() * size_of_head);
                    let mut v_colmaj = Vec::with_capacity(ids.len() * size_of_head);

                    for col in 0..size_of_head {
                        for row in 0..ids.len() {
                            q_colmaj
                                .push(*x4.get(&[row, 0 * n_embd + size_of_head * i_head + col])?);
                            k_colmaj
                                .push(*x4.get(&[row, 1 * n_embd + size_of_head * i_head + col])?);
                            v_colmaj
                                .push(*x4.get(&[row, 2 * n_embd + size_of_head * i_head + col])?);
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
                let x5 = k.transpose(&[1, 0], backend)?;

                // √size_of_head
                let x6 = TypedTensor::<f32>::from_vec_col_major(
                    vec![1, 1],
                    vec![(size_of_head as f32).sqrt()],
                )?;

                // q @ kᵀ / √size_of_head
                let x7 = q.matmul(&x5, backend)?.div(&x6, backend)?;

                //      ⎛ a₁₁ a₁₂ a₁₃ ……… a₁ₙ ⎞        ⎛ a₁₁  −∞  −∞ ………  −∞ ⎞
                //      ⎜ a₂₁ a₂₂ a₂₃ ……… a₂ₙ ⎟        ⎜ a₂₁ a₂₂  −∞ ………  −∞ ⎟
                // x7 = ⎜ a₃₁ a₃₂ a₃₃ ……… a₃ₙ ⎟ → x8 = ⎜ a₃₁ a₃₂ a₃₃ ………  −∞ ⎟
                //      ⎜ ……… ……… ……… ……… ……… ⎟        ⎜ ……… ……… ……… ……… ……… ⎟
                //      ⎝ aₙ₁ aₙ₂ aₙ₃ ……… aₙₙ ⎠        ⎝ aₙ₁ aₙ₂ aₙ₃ ……… aₙₙ ⎠
                let x8 = {
                    let mut colmaj = Vec::with_capacity(ids.len() * ids.len());
                    for col in 0..ids.len() {
                        for row in 0..ids.len() {
                            colmaj.push(if col <= row {
                                *x7.get(&[row, col])?
                            } else {
                                f32::NEG_INFINITY
                            })
                        }
                    }
                    TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), ids.len()], colmaj)?
                };

                // max(x8)
                let x9 = {
                    let mut colmaj = Vec::with_capacity(ids.len());
                    for row in 0..ids.len() {
                        let mut max = f32::NEG_INFINITY;
                        for col in 0..ids.len() {
                            let value = *x8.get(&[row, col])?;
                            if value > max {
                                max = value;
                            }
                        }
                        colmaj.push(max);
                    }
                    TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), 1], colmaj)?
                };

                // x8 - max(x8)
                let x10 = x8.sub(&x9, backend)?;

                // exp(x8 - max(x8))
                let x11 = x10.exp(backend)?;

                // ∑ exp(x8 - max(x8)))
                let x12 = x11
                    .reduce_sum(&[1], backend)?
                    .reshape(&[ids.len(), 1], backend)?;

                // softmax(x8) = exp(x8 - max(x8)) / ∑ exp(x8 - max(x8))
                let x13 = x11.div(&x12, backend)?;

                // softmax(x8) @ v
                let x14 = x13.matmul(&v, backend)?;

                stacked_colmaj.extend(x14.as_slice()?);
            }
            TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], stacked_colmaj)?
        };

        let attn_c_proj_weight = &tensors[&format!("h.{i_layer}.attn.c_proj.weight")];
        if attn_c_proj_weight.shape() != &[n_embd, n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let attn_c_proj_bias = &tensors[&format!("h.{i_layer}.attn.c_proj.bias")];
        if attn_c_proj_bias.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x15 @ attn_c_proj_weight + attn_c_proj_bias
        let x16 = x15
            .matmul(&attn_c_proj_weight, backend)?
            .add(&attn_c_proj_bias, backend)?;

        // x2 + x16
        let x17 = x2.add(&x16, backend).unwrap();

        // ==== Feed Forward ====

        let ln_2_weight = &tensors[&format!("h.{i_layer}.ln_2.weight")];
        if ln_2_weight.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let ln_2_bias = &tensors[&format!("h.{i_layer}.ln_2.bias")];
        if ln_2_bias.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let x18 = layer_norm(&x17, ln_2_weight, ln_2_bias, backend)?;

        let mlp_c_fc_weight = &tensors[&format!("h.{i_layer}.mlp.c_fc.weight")];
        if mlp_c_fc_weight.shape() != &[n_embd, 4 * n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let mlp_c_fc_bias = &tensors[&format!("h.{i_layer}.mlp.c_fc.bias")];
        if mlp_c_fc_bias.shape() != &[4 * n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x18 @ mlp_c_fc_weight + mlp_c_fc_bias
        let x19 = x18
            .matmul(&mlp_c_fc_weight, backend)?
            .add(&mlp_c_fc_bias, backend)?;

        // The formula for GELU is according to the original paper of GELU.
        // https://arxiv.org/pdf/1606.08415

        // 0.044715
        let x20 = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![0.044715])?;

        // √(2 / π)
        let x21 = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![(2.0 / PI).sqrt()])?;

        // 1
        let x22 = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![1.0])?;

        // 0.5
        let x23 = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![0.5])?;

        // GELU(x19) = (tanh((x19³ * 0.044715 + x19) * √(2 / π)) + 1) * x19 * 0.5
        let x24 = x19
            .mul(&x19, backend)?
            .mul(&x19, backend)?
            .mul(&x20, backend)?
            .add(&x19, backend)?
            .mul(&x21, backend)?
            .tanh(backend)?
            .add(&x22, backend)?
            .mul(&x19, backend)?
            .mul(&x23, backend)?;

        let mlp_c_proj_weight = &tensors[&format!("h.{i_layer}.mlp.c_proj.weight")];
        if mlp_c_proj_weight.shape() != &[4 * n_embd, n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        let mlp_c_proj_bias = &tensors[&format!("h.{i_layer}.mlp.c_proj.bias")];
        if mlp_c_proj_bias.shape() != &[n_embd] {
            return Err("tensor has unexpected shape".into());
        }

        // x24 @ mlp_c_proj_weight + mlp_c_proj_bias
        let x25 = x24
            .matmul(&mlp_c_proj_weight, backend)?
            .add(&mlp_c_proj_bias, backend)?;

        // x17 + x25
        x2 = x17.add(&x25, backend).unwrap();
    }

    // ==== Projection ====

    let ln_f_weight = &tensors["ln_f.weight"];
    if ln_f_weight.shape() != &[n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    let ln_f_bias = &tensors["ln_f.bias"];
    if ln_f_bias.shape() != &[n_embd] {
        return Err("tensor has unexpected shape".into());
    }

    let x26 = layer_norm(&x2, ln_f_weight, ln_f_bias, backend)?;

    // x26 @ wte_weightᵀ
    let x27 = x26.matmul(transposed_wte_weight, backend)?;

    Ok(x27)
}

fn layer_norm(
    tensor: &TypedTensor<f32>,
    weight: &TypedTensor<f32>,
    bias: &TypedTensor<f32>,
    backend: &mut tenferro_cpu::CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // N(tensor)
    let x0 = TypedTensor::<f32>::from_vec_col_major(vec![1], vec![tensor.shape()[1] as f32])?;

    // ⟨tensor⟩ = ∑ tensor / N(tensor)
    let x1 = tensor
        .reduce_sum(&[1], backend)?
        .reshape(&[tensor.shape()[0], 1], backend)?
        .div(&x0, backend)?;

    // tensor - ⟨tensor⟩
    let x2 = tensor.sub(&x1, backend)?;

    // The original paper of linear normalization uses N(tensor) for denominator of variance.
    // https://arxiv.org/pdf/1607.06450

    // var(tensor) = ∑ (tensor - ⟨tensor⟩)² / N(tensor)
    let x3 = x2
        .mul(&x2, backend)?
        .reduce_sum(&[1], backend)?
        .reshape(&[tensor.shape()[0], 1], backend)?
        .div(&x0, backend)?;

    // PyTorch uses ε = 1.0e-5.
    // https://docs.pytorch.org/docs/2.13/generated/torch.nn.LayerNorm.html

    // ε = 1.0e-5
    let x4 = TypedTensor::<f32>::from_vec_col_major(vec![1, 1], vec![1.0e-5])?;

    // √(var(tensor) + ε)
    let x5 = x3.add(&x4, backend)?.sqrt(backend)?;

    // (tensor - ⟨tensor⟩) / √(var(tensor) + ε) * weight + bias
    let x6 = x2
        .div(&x5, backend)?
        .mul(&weight, backend)?
        .add(&bias, backend)?;

    Ok(x6)
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
