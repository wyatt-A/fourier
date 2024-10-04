use std::{ffi::{c_char, CStr, CString}, fs::File, io::Read, path::Path, ptr, sync::Mutex};
use cfl::{ndarray::{self, parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator}}, num_complex};
use fftw_bindings::{execute, free_c_string};
use ndarray::{Array3, ArrayD, Ix3, ShapeBuilder};
use num_complex::Complex32;
use std::sync::LazyLock;
use crate::fftw_bindings;

// global FFTW wisdom with thread-safe locking. This happens once at the start of the program
static FFTW_WISDOM: LazyLock<Mutex<String>> = LazyLock::new(||{
    let mut wisdom_string = String::new();
    if let Ok(wks_cache) = std::env::var("WKS_CACHE") {
        let wisdom_file = Path::new(&wks_cache).join("fftw").join("wisdom.dat");
        if let Ok(mut f) = File::open(wisdom_file) {
            if let Err(_) = f.read_to_string(&mut wisdom_string) {
                wisdom_string.clear();
            }else {
                println!("successfully loaded fft wisdom from cache");
            }
        }
    }
    if wisdom_string.is_empty() {
        println!("failed to load fft wisdom from cache!");
    }
    Mutex::new(wisdom_string)
});

#[derive(Clone, Copy)]
enum FFTNorm {
    /// normalize F^H x by N (equivalent to MATLAB)
    Standard,
    /// energy (norm squared) conservation between F x and F^H x
    Unitary,
}

#[derive(Clone, Copy)]
enum FFTDirection {
    Forward,
    Inverse,
}

impl FFTDirection {
    fn to_primitive(&self) -> i32 {
        match &self  {
            // FFTW convention for forward / inverse DFTs
            FFTDirection::Forward => -1,
            FFTDirection::Inverse => 1,
        }
    }
}

pub fn fft3(x:&Array3<Complex32>) -> Array3<Complex32> {
    let mut in_out = x.clone().into_dyn();
    fftn_mut(&mut in_out);
    return in_out.into_dimensionality::<Ix3>().unwrap()
}

pub fn ifft3(x:&Array3<Complex32>) -> Array3<Complex32> {
    let mut in_out = x.clone().into_dyn();
    ifftn_mut(&mut in_out);
    return in_out.into_dimensionality::<Ix3>().unwrap()
}

pub fn fft3_mut(x:&mut Array3<Complex32>) {
    fft3_base(x, FFTDirection::Forward, FFTNorm::Standard);
}

pub fn ifft3_mut(x:&mut Array3<Complex32>) {
    fft3_base(x, FFTDirection::Inverse, FFTNorm::Standard);
}

pub fn fftn(x:&ArrayD<Complex32>) -> ArrayD<Complex32> {
    let mut in_out = x.clone();
    fftn_mut(&mut in_out);
    return in_out
}

pub fn ifftn(x:&ArrayD<Complex32>) -> ArrayD<Complex32> {
    let mut in_out = x.clone();
    ifftn_mut(&mut in_out);
    return in_out
}

pub fn fftn_mut(x:&mut ArrayD<Complex32>) {
    fftn_base(x, FFTDirection::Forward, FFTNorm::Standard);
}

pub fn ifftn_mut(x:&mut ArrayD<Complex32>) {
    fftn_base(x, FFTDirection::Inverse, FFTNorm::Standard);
}

pub fn fftn_mut_unitary(x:&mut ArrayD<Complex32>) {
    fftn_base(x, FFTDirection::Forward, FFTNorm::Unitary);
}

pub fn ifftn_mut_unitary(x:&mut ArrayD<Complex32>) {
    fftn_base(x, FFTDirection::Inverse, FFTNorm::Unitary);
}

fn fftn_base(x:&mut ArrayD<Complex32>, direction:FFTDirection, norm:FFTNorm) {

    let mut input_shape = x.shape().to_owned();
    input_shape.reverse();
    let rank = input_shape.len();

    if !is_fortran_order(x.shape(),x.strides()) {
        let mut tmp = ArrayD::<Complex32>::zeros(x.shape().f());
        tmp.assign(&x);
        let data = tmp.as_slice_memory_order_mut().unwrap();
        fftn_exec_inplace(rank, input_shape.as_slice(), direction, data );
        fft_normalize(data, data.len(), norm, direction);
        x.assign(&tmp);
    }else {
        let data = x.as_slice_memory_order_mut().unwrap();
        fftn_exec_inplace(rank, input_shape.as_slice(), direction, data);
        fft_normalize(data, data.len(), norm, direction);
    }

}

fn fft3_base(x:&mut Array3<Complex32>, direction:FFTDirection, norm:FFTNorm) {

    let mut input_shape = x.shape().to_owned();
    input_shape.reverse();
    let rank = input_shape.len();

    if !is_fortran_order(x.shape(),x.strides()) {
        let mut tmp = Array3::<Complex32>::zeros(x.dim().f());
        tmp.assign(&x);
        let data = tmp.as_slice_memory_order_mut().unwrap();
        fftn_exec_inplace(rank, input_shape.as_slice(), direction, data );
        fft_normalize(data, data.len(), norm, direction);
        x.assign(&tmp);
    }else {
        let data = x.as_slice_memory_order_mut().unwrap();
        fftn_exec_inplace(rank, input_shape.as_slice(), direction, data);
        fft_normalize(data, data.len(), norm, direction);
    }

}


fn fft_normalize(x:&mut [Complex32],n:usize,norm:FFTNorm,direction:FFTDirection) {
    match norm {
        FFTNorm::Standard => {
            // only normalize for inverse xforms
            if let FFTDirection::Inverse = direction {
                let scale = 1. / n as f32;
                x.par_iter_mut().for_each(|x| *x *= scale);
            }
        },
        FFTNorm::Unitary => {
            // scale regardless of direction
            let scale = 1. / (n as f32).sqrt();
            x.par_iter_mut().for_each(|x| *x *= scale);
        }
    }
}

fn fftn_exec_inplace(
    rank: usize,
    n: &[usize],
    sign: FFTDirection,
    in_out: &mut [Complex32],
) {

    let sign = sign.to_primitive();
    let rank = rank as i32;
    let n:Vec<i32> = n.iter().map(|&n|n as i32).collect();
    let ptr = in_out.as_mut_ptr();

    let mut ws = FFTW_WISDOM.lock().unwrap();

    let c_wisdom_string_in = CString::new(ws.as_str()).expect("CString::new failed");
    let c_wisdom_string_in_ptr: *const c_char = c_wisdom_string_in.as_ptr();

    let mut c_wisdom_string_out: *mut c_char = ptr::null_mut();

    let mut status = unsafe {
        execute(
            rank,
            n.as_ptr(),
            sign,
            c_wisdom_string_in_ptr,
            &mut c_wisdom_string_out as *mut *mut c_char,
            ptr,
            ptr,
        )
    };

    if c_wisdom_string_out.is_null() {
        panic!("string export failed");
    }

    unsafe {
        let c_str = CStr::from_ptr(c_wisdom_string_out)
            .to_str()                 
            .expect("Invalid UTF-8");
        ws.clear();
        ws.push_str(c_str);
        free_c_string(c_wisdom_string_out);
    };

    // we need to try again ...
    if status == -1 {

        let c_wisdom_string_in = CString::new(ws.as_str()).expect("CString::new failed");
        let c_wisdom_string_in_ptr: *const c_char = c_wisdom_string_in.as_ptr();
        let mut c_wisdom_string_out: *mut c_char = ptr::null_mut();

        status = unsafe {
            execute(
                rank,
                n.as_ptr(),
                sign,
                c_wisdom_string_in_ptr,
                &mut c_wisdom_string_out as *mut *mut c_char,
                ptr,
                ptr,
            )
        };
        unsafe {
            free_c_string(c_wisdom_string_out);
        };
    }
    assert_eq!(status,0,"fftn failed with error code {}",status);
}


/// checks that array shape and strides are in column-major (fortran) order
fn is_fortran_order(shape:&[usize],strides:&[isize]) -> bool {
    // Check if strides correspond to Fortran ordering (column-major)
    if shape.len() < 2 {
        // For a 1D array, we consider it to be in Fortran order by default.
        return true;
    }
    for i in 0..shape.len() - 1 {
        if strides[i] < strides[i + 1] {
            return false;
        }
    }
    true
}