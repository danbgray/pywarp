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

#[pyfunction]
fn ricci_t_loops<'py>(py: Python<'py>, diff1: PyReadonlyArrayDyn<'_, f64>, diff2: PyReadonlyArrayDyn<'_, f64>, inv: PyReadonlyArrayDyn<'_, f64>) -> PyResult<&'py PyArrayDyn<f64>> {
    let diff1 = diff1.as_array();
    let diff2 = diff2.as_array();
    let inv = inv.as_array();

    if diff1.ndim() != 4 || diff2.ndim() != 5 || inv.ndim() != 3
        || diff1.shape()[0] != 4 || diff1.shape()[1] != 4 || diff1.shape()[2] != 4
        || diff2.shape()[0] != 4 || diff2.shape()[1] != 4 || diff2.shape()[2] != 4 || diff2.shape()[3] != 4
        || inv.shape()[0] != 4 || inv.shape()[1] != 4
    {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid tensor dimensions"));
    }
    let n = diff1.shape()[3];
    let mut ricci = Array3::<f64>::zeros((4,4,n));
    for i in 0..4 {
        for j in i..4 {
            for idx in 0..n {
                let mut temp = 0f64;
                for a in 0..4 {
                    for b in 0..4 {
                        temp -= 0.5 * (
                            diff2[[i,j,a,b,idx]] +
                            diff2[[a,b,i,j,idx]] -
                            diff2[[i,b,j,a,idx]] -
                            diff2[[j,b,i,a,idx]]
                        ) * inv[[a,b,idx]];
                    }
                }
                for a in 0..4 {
                    for b in 0..4 {
                        for c in 0..4 {
                            for d in 0..4 {
                                temp += 0.5 * (
                                    0.5 * diff1[[a,c,i,idx]] * diff1[[b,d,j,idx]] +
                                    diff1[[i,c,a,idx]] * diff1[[j,d,b,idx]] -
                                    diff1[[i,c,a,idx]] * diff1[[j,b,d,idx]]
                                ) * inv[[a,b,idx]] * inv[[c,d,idx]];
                                temp -= 0.25 * (
                                    diff1[[j,c,i,idx]] + diff1[[i,c,j,idx]] - diff1[[i,j,c,idx]]
                                ) * (2.0 * diff1[[b,d,a,idx]] - diff1[[a,b,d,idx]])
                                    * inv[[a,b,idx]] * inv[[c,d,idx]];
                            }
                        }
                    }
                }
                ricci[[i,j,idx]] = temp;
                if i != j {
                    ricci[[j,i,idx]] = temp;
                }
            }
        }
    }
    Ok(PyArrayDyn::from_owned_array(py, ricci.into_dyn()))
}

#[pymodule]
fn warp_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(c4_inv, m)?)?;
    m.add_function(wrap_pyfunction!(ricci_t_loops, m)?)?;
    Ok(())
}
