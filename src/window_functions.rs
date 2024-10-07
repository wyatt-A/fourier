// use std::{f32::consts::PI, process::Output};

// use ndarray::{ArrayD, Dimension};
// use num_complex::Complex32;
// use num_traits::ToPrimitive;

// use crate::{apply_window, subscript_to_freq_bin};

use std::f32::consts::PI;

use cfl::{ndarray::{ArrayD, Dimension, ShapeBuilder}, num_complex::Complex32};
use rustfft::num_traits::ToPrimitive;

use crate::rustfft::subscript_to_freq_bin;

pub fn tukey_win(r: f32, r1: f32, r2: f32, min: f32, max: f32) -> f32 {
    let val = if r < r1 {
        1.
    } else if r >= r1 && r < r2 {
        (PI * (r - r1) / (r2 - r1)).cos() * 0.5 + 0.5
    } else {
        0.
    };

    val * (max - min) + min
}

pub struct TukeyWindow {
    r1:f32,
    r2:f32,
    min:f32,
    max:f32,
}

impl TukeyWindow {
    /// r1 is the inner radius where there is no roll-off,
    /// r2 is the outer radius where the roll-off is complete,
    /// min is the value outside of r2,
    /// max is the value insize of r1
    pub fn new(r1:f32,r2:f32,min:f32,max:f32) -> Self {
        Self { r1, r2, min, max }
    }
}

impl WindowFunction for TukeyWindow {
    type OutputScalar = f32;
    fn evaluate<T:ToPrimitive>(&self,coord:&[T]) -> Self::OutputScalar {
        let r = coord.iter()
        .map(|t|t.to_f32().expect("failed to convert to float"))
        .map(|x|x.powi(2))
        .sum::<f32>()
        .sqrt();
        let val = if r < self.r1 {
            1.
        } else if r >= self.r1 && r < self.r2 {
            (PI * (r - self.r1) / (self.r2 - self.r1)).cos() * 0.5 + 0.5
        } else {
            0.
        };
        val * (self.max - self.min) + self.min
    }
}



pub struct HanningWindow {
    window_size:Vec<usize>
}

impl HanningWindow {
    pub fn new(window_size:&[usize]) -> Self {
        Self {
            window_size: window_size.to_owned()
        }
    }
}

impl WindowFunction for HanningWindow {
    type OutputScalar = f32;

    fn evaluate<T:ToPrimitive>(&self,coord:&[T]) -> Self::OutputScalar {
        self.window_size.iter().zip(coord.iter()).map(|(d,c)|{
            0.5 * ((2.0 * PI * c.to_f32().unwrap() / *d as f32).cos() + 1.)
        }).product()
    }
    
}

pub trait WindowFunction {
    type OutputScalar: ToPrimitive;
    fn evaluate<T:ToPrimitive>(&self,coord:&[T]) -> Self::OutputScalar;
    fn window(&self,size:&[usize]) -> ArrayD<f32> {
        let n_dims = size.len();
        let dims = size.to_owned();
        let mut freq_bin = vec![0; n_dims];
        let mut idxs = vec![0usize; n_dims];
        let mut kern = ArrayD::from_elem(size.f(), 0.);
        kern.indexed_iter_mut().for_each(|(idx, val)| {
            idxs.iter_mut()
                .zip(idx.as_array_view().iter())
                .for_each(|(a, b)| *a = *b);
            subscript_to_freq_bin(&dims, &idxs, &mut freq_bin);
            *val = self.evaluate(&freq_bin).to_f32().expect("failed to convert to float");
        });
        kern
    }
    fn apply(&self,x:&mut ArrayD<Complex32>) {
        let n_dims = x.shape().len();
        let dims = x.shape().to_owned();
        let mut freq_bin = vec![0; n_dims];
        let mut idxs = vec![0usize; n_dims];
        x.indexed_iter_mut().for_each(|(idx, val)| {
            idxs.iter_mut()
                .zip(idx.as_array_view().iter())
                .for_each(|(a, b)| *a = *b);
            subscript_to_freq_bin(&dims, &idxs, &mut freq_bin);
            let w = self.evaluate(&freq_bin).to_f32().expect("failed to convert to float");
            *val *= w;
        });
    }
}
