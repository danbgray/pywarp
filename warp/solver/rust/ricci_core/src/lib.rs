use numpy::{PyArray3, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

#[pyfunction]
fn ricci_t_loops<'py>(py: Python<'py>, diff1: PyReadonlyArrayDyn<'py, f64>, diff2: PyReadonlyArrayDyn<'py, f64>, inv: PyReadonlyArrayDyn<'py, f64>) -> &'py PyArray3<f64> {
    let diff1 = diff1.as_array();
    let diff2 = diff2.as_array();
    let inv = inv.as_array();
    let size = diff1.shape()[3];
    let ricci = PyArray3::<f64>::zeros(py, [4, 4, size], false);
    let mut ricci_mut = unsafe { ricci.as_array_mut() };
    for i in 0..4 {
        for j in i..4 {
            for idx in 0..size {
                let mut temp = 0.0;
                for a in 0..4 {
                    for b in 0..4 {
                        temp -= 0.5 * (
                            diff2[[i, j, a, b, idx]]
                                + diff2[[a, b, i, j, idx]]
                                - diff2[[i, b, j, a, idx]]
                                - diff2[[j, b, i, a, idx]])
                            * inv[[a, b, idx]];
                    }
                }
                for a in 0..4 {
                    for b in 0..4 {
                        for c in 0..4 {
                            for d in 0..4 {
                                temp += 0.5
                                    * (0.5 * diff1[[a, c, i, idx]] * diff1[[b, d, j, idx]]
                                        + diff1[[i, c, a, idx]] * diff1[[j, d, b, idx]]
                                        - diff1[[i, c, a, idx]] * diff1[[j, b, d, idx]])
                                    * inv[[a, b, idx]]
                                    * inv[[c, d, idx]];
                                temp -= 0.25
                                    * (diff1[[j, c, i, idx]] + diff1[[i, c, j, idx]] - diff1[[i, j, c, idx]])
                                    * (2.0 * diff1[[b, d, a, idx]] - diff1[[a, b, d, idx]])
                                    * inv[[a, b, idx]]
                                    * inv[[c, d, idx]];
                            }
                        }
                    }
                }
                ricci_mut[[i, j, idx]] = temp;
                if i != j {
                    ricci_mut[[j, i, idx]] = temp;
                }
            }
        }
    }
    ricci
}

#[pymodule]
fn ricci_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ricci_t_loops, m)?)?;
    Ok(())
}
