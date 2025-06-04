use ndarray::prelude::*;
use ndarray::{s, Zip};
use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn, PyArrayDyn};
use nalgebra::Matrix4;

#[pyfunction]
fn c4_inv<'py>(py: Python<'py>, tensor: PyReadonlyArrayDyn<'_, f64>) -> PyResult<&'py PyArrayDyn<f64>> {
    let tensor = tensor.as_array();
    // check shape
    if tensor.shape().len() < 2 || tensor.shape()[0] != 4 || tensor.shape()[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("The first two dimensions of the input tensor must be of size 4."));
    }
    let orig_shape = tensor.shape().to_vec();
    let extra_dims = &orig_shape[2..];
    let total: usize = extra_dims.iter().product();
    let tensor = tensor.view().into_shape((4, 4, total)).unwrap();
    let mut result = ArrayD::<f64>::zeros(IxDyn(&orig_shape));
    {
        let mut out_view = result.view_mut().into_shape((4, 4, total)).unwrap();
        for i in 0..total {
            let slice = tensor.slice(s![.., .., i]);
            let owned = slice.to_owned();
            let mat = Matrix4::from_row_slice(owned.as_slice().unwrap());
            if let Some(inv) = mat.try_inverse() {
                let mut inv_slice = out_view.slice_mut(s![.., .., i]);
                inv_slice.assign(&ndarray::Array2::from_shape_vec((4, 4), inv.as_slice().to_vec()).unwrap());
            } else {
                let mut inv_slice = out_view.slice_mut(s![.., .., i]);
                inv_slice.assign(&ndarray::Array2::eye(4));
            }
        }
    }
    Ok(PyArrayDyn::from_owned_array(py, result))
}

fn diff1(arr: &ArrayD<f64>, axis: usize, delta: f64) -> ArrayD<f64> {
    let mut result = ArrayD::<f64>::zeros(arr.raw_dim());
    let n = arr.shape()[axis];
    for i in 0..n {
        let mut out_slice = result.index_axis_mut(Axis(axis), i);
        if i == 0 {
            let next = arr.index_axis(Axis(axis), i + 1);
            let curr = arr.index_axis(Axis(axis), i);
            Zip::from(&mut out_slice).and(&next).and(&curr).apply(|o, &nxt, &cur| {
                *o = (nxt - cur) / delta;
            });
        } else if i == n - 1 {
            let curr = arr.index_axis(Axis(axis), i);
            let prev = arr.index_axis(Axis(axis), i - 1);
            Zip::from(&mut out_slice).and(&curr).and(&prev).apply(|o, &cur, &prv| {
                *o = (cur - prv) / delta;
            });
        } else {
            let next = arr.index_axis(Axis(axis), i + 1);
            let prev = arr.index_axis(Axis(axis), i - 1);
            Zip::from(&mut out_slice).and(&next).and(&prev).apply(|o, &nxt, &prv| {
                *o = (nxt - prv) / (2.0 * delta);
            });
        }
    }
    result
}

#[pyfunction]
fn take_finite_difference1<'py>(
    py: Python<'py>,
    tensor: PyReadonlyArrayDyn<'_, f64>,
    axis: usize,
    delta: Vec<f64>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = tensor.as_array().to_owned();
    if axis >= arr.ndim() || axis >= delta.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Axis out of range"));
    }
    let res = diff1(&arr, axis, delta[axis]);
    Ok(PyArrayDyn::from_owned_array(py, res))
}

#[pyfunction]
fn take_finite_difference2<'py>(
    py: Python<'py>,
    tensor: PyReadonlyArrayDyn<'_, f64>,
    axis1: usize,
    axis2: usize,
    delta: Vec<f64>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let arr = tensor.as_array().to_owned();
    if axis1 >= arr.ndim() || axis2 >= arr.ndim() || axis1 >= delta.len() || axis2 >= delta.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Axis out of range"));
    }
    let first = diff1(&arr, axis1, delta[axis1]);
    let second = diff1(&first, axis2, delta[axis2]);
    Ok(PyArrayDyn::from_owned_array(py, second))
}

#[pymodule]
fn warp_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(c4_inv, m)?)?;
    m.add_function(wrap_pyfunction!(take_finite_difference1, m)?)?;
    m.add_function(wrap_pyfunction!(take_finite_difference2, m)?)?;
    Ok(())
}
