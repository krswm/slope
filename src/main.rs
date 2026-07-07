// My first tenferro
// https://tensor4all.org/tenferro-rs/getting-started/#first-cpu-program

// It seems I need this in order to a.matmul
use tenferro_runtime::TensorOpsExt;

use tenferro_runtime::TypedTensor;

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

    println!("----");

    // I need transpose

    let d = tenferro_runtime::Tensor::from_vec_col_major(
        vec![2, 3], vec![1.0, 3.0, 2.0, 4.0, 10.0, 20.0]
    )?;
    println!("{:?}", d.as_slice::<f64>().unwrap());
    let e = d.transpose(&[1, 0], &mut backend)?;
    println!("{:?}", e.as_slice::<f64>().unwrap());

    println!("----");

    // I need matrix addition

    // https://tensor4all.org/tenferro-rs/spec/operation-categories.html#elementwise-arithmetic-comparison-selection 

    let f = tenferro_runtime::Tensor::from_vec_col_major(
        vec![2, 3], vec![1.0, 3.0, 2.0, 4.0, -1.0, -2.0]
    )?;
    let g = tenferro_runtime::Tensor::from_vec_col_major(
        vec![2, 3], vec![0.1, 0.2, 0.2, 0.4, -0.1, -0.2]
    )?;

    let h = f.add(&g, &mut backend)?;
    println!("{:?}", h.as_slice::<f64>().unwrap());

    println!("----");

    // I need upper triangular matrix

    // https://tensor4all.org/tenferro-rs/spec/operation-categories.html#construction

    // I may have to use TypedTensor instead of Tensor
    // since I will use only float32 tensors

    // T is type but what is R?
    // https://tensor4all.org/tenferro-rs/api/tenferro_tensor/types/struct.TypedTensor.html#method.ones

    let i = TypedTensor::<f32>::ones(vec![2, 3])?;
    println!("{:?}", i.host_data()?);

    // Eager? Traced?

    // let j = i.triu(&mut backend)?;
    // println!("{:?}", j.host_data()?);

    Ok(())
}
