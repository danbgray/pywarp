use ndarray::prelude::*;
use ndarray::s;
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

#[pymodule]
fn warp_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(c4_inv, m)?)?;
    Ok(())
}
