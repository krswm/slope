// My first tenferro
// https://tensor4all.org/tenferro-rs/getting-started/#first-cpu-program

// It seems I need this in order to a.matmul
use tenferro_runtime::TensorOpsExt;

// What is Box?
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = tenferro_cpu::CpuBackend::new();

    let a = tenferro_runtime::Tensor::from_vec_col_major(
        vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]
    )?;
    let b = tenferro_runtime::Tensor::from_vec_col_major(
        vec![2, 2], vec![5.0, 7.0, 6.0, 8.0]
    )?;

    // a @ b?
    let c = a.matmul(&b, &mut backend)?;

    println!("{:?}", c.shape());
    println!("{:?}", c.as_slice::<f64>().unwrap());

    Ok(())
}
