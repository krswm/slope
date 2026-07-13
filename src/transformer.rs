use std::collections::HashMap;
use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub struct SetUp {
    pub wte_weight: TypedTensor<f32>,
    pub wpe_weight: TypedTensor<f32>,
    pub ln_f_weight: TypedTensor<f32>,
    pub ln_f_bias: TypedTensor<f32>,
    pub ln_1_weight: Vec<TypedTensor<f32>>,
    pub ln_1_bias: Vec<TypedTensor<f32>>,
    pub attn_c_attn_weight: Vec<TypedTensor<f32>>,
    pub attn_c_attn_bias: Vec<TypedTensor<f32>>,
    pub attn_c_proj_weight: Vec<TypedTensor<f32>>,
    pub attn_c_proj_bias: Vec<TypedTensor<f32>>,
    pub ln_2_weight: Vec<TypedTensor<f32>>,
    pub ln_2_bias: Vec<TypedTensor<f32>>,
    pub mlp_c_fc_weight: Vec<TypedTensor<f32>>,
    pub mlp_c_fc_bias: Vec<TypedTensor<f32>>,
    pub mlp_c_proj_weight: Vec<TypedTensor<f32>>,
    pub mlp_c_proj_bias: Vec<TypedTensor<f32>>,
}

pub fn get_setup(
    tensors: std::collections::HashMap<String, TypedTensor<f32>>,
    n_ctx: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    vocab_size: usize,
    backend: &mut CpuBackend,
) -> Result<SetUp, Box<dyn Error>> {
    let wte_weight = *tensors.get("wte.weight").unwrap();
    let wpe_weight = *tensors.get("wpe.weight").unwrap();
    let ln_f_weight = *tensors.get("ln_f.weight").unwrap();
    let ln_f_bias = *tensors.get("ln_f.bias").unwrap();

    let mut ln_1_weight = Vec::new();
    let mut ln_1_bias = Vec::new();
    let mut attn_c_attn_weight = Vec::new();
    let mut attn_c_attn_bias = Vec::new();
    let mut attn_c_proj_weight = Vec::new();
    let mut attn_c_proj_bias = Vec::new();
    let mut ln_2_weight = Vec::new();
    let mut ln_2_bias = Vec::new();
    let mut mlp_c_fc_weight = Vec::new();
    let mut mlp_c_fc_bias = Vec::new();
    let mut mlp_c_proj_weight = Vec::new();
    let mut mlp_c_proj_bias = Vec::new();

    for i_layer in 0..n_layer {
        ln_1_weight.push(*tensors.get(&format!("h.{i_layer}.ln_1.weight")).unwrap());
        ln_1_bias.push(*tensors.get(&format!("h.{i_layer}.ln_1.bias")).unwrap());
        attn_c_attn_weight.push(
            *tensors
                .get(&format!("h.{i_layer}.attn.c_attn.weight"))
                .unwrap(),
        );
        attn_c_attn_bias.push(
            *tensors
                .get(&format!("h.{i_layer}.attn.c_attn.bias"))
                .unwrap(),
        );
        attn_c_proj_weight.push(
            *tensors
                .get(&format!("h.{i_layer}.attn.c_proj.weight"))
                .unwrap(),
        );
        attn_c_proj_bias.push(
            *tensors
                .get(&format!("h.{i_layer}.attn.c_proj.bias"))
                .unwrap(),
        );
        ln_2_weight.push(*tensors.get(&format!("h.{i_layer}.ln_2.weight")).unwrap());
        ln_2_bias.push(*tensors.get(&format!("h.{i_layer}.ln_2.bias")).unwrap());
        mlp_c_fc_weight.push(
            *tensors
                .get(&format!("h.{i_layer}.mlp.c_fc.weight"))
                .unwrap(),
        );
        mlp_c_fc_bias.push(*tensors.get(&format!("h.{i_layer}.mlp.c_fc.bias")).unwrap());
        mlp_c_proj_weight.push(
            *tensors
                .get(&format!("h.{i_layer}.mlp.c_proj.weight"))
                .unwrap(),
        );
        mlp_c_proj_bias.push(
            *tensors
                .get(&format!("h.{i_layer}.mlp.c_proj.bias"))
                .unwrap(),
        );
    }

    Ok(SetUp {
        wte_weight: wte_weight,
        wpe_weight: wpe_weight,
        ln_f_weight: ln_f_weight,
        ln_f_bias: ln_f_bias,
        ln_1_weight: ln_1_weight,
        ln_1_bias: ln_1_bias,
        attn_c_attn_weight: attn_c_attn_weight,
        attn_c_attn_bias: attn_c_attn_bias,
        attn_c_proj_weight: attn_c_proj_weight,
        attn_c_proj_bias: attn_c_proj_bias,
        ln_2_weight: ln_2_weight,
        ln_2_bias: ln_2_bias,
        mlp_c_fc_weight: mlp_c_fc_weight,
        mlp_c_fc_bias: mlp_c_fc_bias,
        mlp_c_proj_weight: mlp_c_proj_weight,
        mlp_c_proj_bias: mlp_c_proj_bias,
    })
}

pub fn transform(
    wte_weight_transposed: &TypedTensor<f32>,
    n_ctx: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    vocab_size: usize,
    su: &SetUp,
    ids: &Vec<usize>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let mut x = {
        // wte[ids]
        let x0 = {
            let mut colmaj = Vec::with_capacity(ids.len() * n_embd);
            for i in 0..n_embd {
                for id in ids {
                    colmaj.push(*su.wte_weight.get(&[*id, i])?);
                }
            }
            TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], colmaj)?
        };

        // wpe[range(len(ids))]
        let x1 = {
            let mut colmaj = Vec::with_capacity(ids.len() * n_embd);
            for col in 0..n_embd {
                for row in 0..ids.len() {
                    colmaj.push(*su.wpe_weight.get(&[row, col])?);
                }
            }
            TypedTensor::<f32>::from_vec_col_major(vec![ids.len(), n_embd], colmaj)?
        };

        // wte[ids] + wpe[range(len(ids))]
        x0.add(&x1, backend)?
    };

    let n_ids = ids.len();

    for i_layer in 0..n_layer {
        let y0 = {
            layer_norm(
                &x,
                &su.ln_1_weight[i_layer],
                &su.ln_1_bias[i_layer],
                n_embd,
                n_ids,
                backend,
                i_layer == 0,
            )
        };

        // b0 @ attn_c_attn_weight
        let b1 = y0.matmul(&su.attn_c_attn_weight[i_layer], backend)?;

        // b0 @ attn_c_attn_weight + attn_c_attn_bias
        let b2 = b1.add(&su.attn_c_attn_bias[i_layer], backend)?;

        let size_of_head = n_embd / n_head;

        let mut raw_stacked: Vec<f32> = Vec::new();
        for i_head in 0..n_head {
            let (q, k, v) = {
                let mut q_colmaj = Vec::new();
                let mut k_colmaj = Vec::new();
                let mut v_colmaj = Vec::new();

                for col in 0..size_of_head {
                    for row in 0..n_ids {
                        q_colmaj.push(*b2.get(&[row, 0 * n_embd + size_of_head * i_head + col])?);
                        k_colmaj.push(*b2.get(&[row, 1 * n_embd + size_of_head * i_head + col])?);
                        v_colmaj.push(*b2.get(&[row, 2 * n_embd + size_of_head * i_head + col])?);
                    }
                }

                (
                    TypedTensor::<f32>::from_vec_col_major(vec![n_ids, size_of_head], q_colmaj)?,
                    TypedTensor::<f32>::from_vec_col_major(vec![n_ids, size_of_head], k_colmaj)?,
                    TypedTensor::<f32>::from_vec_col_major(vec![n_ids, size_of_head], v_colmaj)?,
                )
            };

            let b3 = k.transpose(&[1, 0], backend)?;

            let b4 = q.matmul(&b3, backend).unwrap();

            let b5 = TypedTensor::<f32>::from_vec_col_major(
                vec![1, 1],
                vec![(size_of_head as f32).sqrt()],
            )?;

            let b6 = b4.div(&b5, backend)?;

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
                    raw_causal_mask.push(if col <= row { 0.0 } else { f32::NEG_INFINITY })
                }
            }
            let causal_mask =
                TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_ids], raw_causal_mask)
                    .unwrap();

            let xii = b6.add(&causal_mask, backend).unwrap();

            let mut raw_max_xii = Vec::new();
            for row in 0..n_ids {
                let mut max = -1.0e12f32;
                for col in 0..n_ids {
                    max = max.max(*xii.get(&[row, col])?);
                }
                raw_max_xii.push(max);
            }
            let max_xii = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, 1], raw_max_xii)?;

            let negative_shift_xii = xii.sub(&max_xii, backend)?;

            let e = negative_shift_xii.exp(backend)?;

            let e_sum = e.reduce_sum(&[1], backend)?.reshape(&[n_ids, 1], backend)?;

            let xj = e.div(&e_sum, backend)?;

            let xk = xj.matmul(&v, backend)?;

            raw_stacked.extend(xk.as_slice()?);
        }
        let stacked = TypedTensor::<f32>::from_vec_col_major(vec![n_ids, n_embd], raw_stacked)?;

        let xl = stacked
            .matmul(&su.attn_c_proj_weight[i_layer], backend)
            .unwrap();

        let xm = xl.add(&su.attn_c_proj_bias[i_layer], backend).unwrap();

        let xn = x.add(&xm, backend).unwrap();

        let xo = layer_norm(
            &xn,
            &su.ln_2_weight[i_layer],
            &su.ln_2_bias[i_layer],
            n_embd,
            n_ids,
            backend,
            false,
        );

        let xp = xo.matmul(&su.mlp_c_fc_weight[i_layer], backend).unwrap();

        let xq = xp.add(&su.mlp_c_fc_bias[i_layer], backend).unwrap();

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

        let xs = xr_scaled
            .matmul(&su.mlp_c_proj_weight[i_layer], backend)
            .unwrap();

        let xt = xs.add(&su.mlp_c_proj_bias[i_layer], backend).unwrap();

        x = xn.add(&xt, backend).unwrap();
    }

    let xu = layer_norm(
        &x,
        &su.ln_f_weight,
        &su.ln_f_bias,
        n_embd,
        n_ids,
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
