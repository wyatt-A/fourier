use array_utils::idx_to_sub_col_major;
use cfl::{ndarray::{ArrayD, Axis, Dimension, ShapeBuilder}, ndarray_stats::QuantileExt, num_complex::Complex32};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rustfft::{num_traits::ToPrimitive, FftDirection, FftPlanner};
use std::error::Error;
use crate::window_functions::WindowFunction;

/// n-d fourier transform
pub fn fftn_par(vol: &mut ArrayD<Complex32>, direction: FftDirection) {
    let dims = vol.shape().to_owned();
    let n: f32 = dims.iter().product::<usize>() as f32;
    let scale = 1. / n.sqrt();

    for ax in 0..dims.len() {
        // we must skip singleton dims, otherwise there is a massive slow-down
        // likely from the .as_standard_layout() call
        if dims[ax] == 1 {
            continue;
        }
        let planner = FftPlanner::<f32>::new().plan_fft(dims[ax], direction);
        vol.lanes_mut(Axis(ax))
            .into_iter()
            .par_bridge()
            .for_each(|mut lane| {
                let mut view = lane.as_standard_layout().to_owned();
                planner.process(view.as_slice_mut().unwrap());
                lane.assign(&view);
            });
    }
    vol.map_inplace(|x| *x *= scale);
}

/// forward n-d unitary fourier transform
pub fn fftn(vol: &mut ArrayD<Complex32>) {
    fftn_par(vol, FftDirection::Forward);
}

/// inverse n-d unitary fourier transform
pub fn ifftn(vol: &mut ArrayD<Complex32>) {
    fftn_par(vol, FftDirection::Inverse);
}

/// forward n-d centered unitary fourier transform
pub fn fftnc(x: &mut ArrayD<Complex32>) {
    ifftshift(x);
    fftn_par(x, FftDirection::Forward);
    fftshift(x);
}

/// inverse n-d centered unitary fourier transform
pub fn ifftnc(x: &mut ArrayD<Complex32>) {
    ifftshift(x);
    fftn_par(x, FftDirection::Inverse);
    fftshift(x);
}

/// n-d fft shift, returning the shift that was applied
pub fn fftshift(vol: &mut ArrayD<Complex32>) -> Vec<i32> {
    let dims = vol.shape();
    let shifts = fftshifts(dims);
    circshift(vol, &shifts);
    shifts
}

/// n-d inverse fft shift, returning the shift that was applied
pub fn ifftshift(vol: &mut ArrayD<Complex32>) -> Vec<i32> {
    let dims = vol.shape();
    let shifts = ifftshifts(dims);
    circshift(vol, &shifts);
    shifts
}

#[inline(always)]
pub fn fftshifts(dims: &[usize]) -> Vec<i32> {
    dims.iter()
        .map(|x| if x % 2 == 0 { x / 2 } else { (x - 1) / 2 } as i32)
        .collect()
}

#[inline(always)]
pub fn ifftshifts(dims: &[usize]) -> Vec<i32> {
    dims.iter()
        .map(|x| if x % 2 == 0 { x / 2 } else { (x + 1) / 2 } as i32)
        .collect()
}

/// n-d circular shift
pub fn circshift(vol: &mut ArrayD<Complex32>, shift: &[i32]) {
    let dims = vol.shape().to_owned();
    if shift.len() != dims.len() {
        panic!("shift must be the same size as the number of array dims");
    }
    for ax in 0..dims.len() {
        if shift[ax] > 0 {
            let sh = shift[ax].unsigned_abs() as usize;
            vol.lanes_mut(Axis(ax))
                .into_iter()
                .par_bridge()
                .for_each(|mut lane| {
                    let mut buff = lane.to_owned();
                    buff.as_slice_mut().unwrap().rotate_right(sh);
                    lane.assign(&buff);
                });
        } else if shift[ax] < 0 {
            let sh = shift[ax].unsigned_abs() as usize;
            vol.lanes_mut(Axis(ax))
                .into_iter()
                .par_bridge()
                .for_each(|mut lane| {
                    let mut buff = lane.to_owned();
                    buff.as_slice_mut().unwrap().rotate_left(sh);
                    lane.assign(&buff);
                });
        }
    }
}

/// finds the subscript of the sample with the highest energy
pub fn _dc_subscript(x: &ArrayD<Complex32>) -> Result<Vec<usize>, Box<dyn Error>> {
    Ok(x.map(|x| x.norm_sqr()).argmax()?.as_array_view().to_vec())
}

/// finds the subscript of the sample with the highest energy. The array must be in contiguous fortran memory
/// order. Returns an error otherwise
pub fn dc_subscript(x: &ArrayD<Complex32>) -> Result<Vec<usize>, Box<dyn Error>> {
    if x.is_standard_layout() {
        Err("array cannot be in c memory layout")?
    }

    let mut max = 0.;
    let mut idx = 0;
    for (i, val) in x.as_slice_memory_order().unwrap().iter().enumerate() {
        let e = val.norm_sqr();
        if e > max {
            max = e;
            idx = i;
        }
    }

    let mut sub = x.shape().to_owned();
    idx_to_sub_col_major(idx, x.shape(), &mut sub)?;

    Ok(sub)
}

/// shift the DC sample to first array index = 0, returning the shift that was performed
pub fn set_dc_to_origin(x: &mut ArrayD<Complex32>) -> Result<Vec<i32>, Box<dyn Error>> {
    let dc_sub = dc_subscript(x)?;
    let shift: Vec<_> = dc_sub.into_iter().map(|x| -(x as i32)).collect();
    circshift(x, &shift);
    Ok(shift)
}

/// shift the DS sample to the center of the array, returning the shift that was performed
pub fn set_dc_to_center(x: &mut ArrayD<Complex32>) -> Result<Vec<i32>, Box<dyn Error>> {
    let mut net_shift = set_dc_to_origin(x)?;
    let fft_shifts = fftshift(x);
    // compute the net shift of operation
    net_shift
        .iter_mut()
        .zip(fft_shifts.iter())
        .for_each(|(x, y)| *x += y);
    Ok(net_shift)
}

/// set the phase of the DC sample, returning the old phase value
pub fn set_dc_phase(x: &mut ArrayD<Complex32>, phase: f32) -> Result<f32, Box<dyn Error>> {
    let dc = dc_subscript(x)?;
    let current_phase = x[dc.as_slice()].to_polar().1;
    let phase_adj = phase - current_phase;
    let r = Complex32::from_polar(1., phase_adj);
    x.par_mapv_inplace(|x| x * r);
    Ok(current_phase)
}

/// assuming that the DC sample is at the origin
pub fn subscript_to_freq_bin(dimensions: &[usize], subscript: &[usize], freq_bin: &mut [i32]) {
    dimensions
        .iter()
        .zip(subscript.iter())
        .zip(freq_bin.iter_mut())
        .for_each(|((d, s), val)| {
            *val = index_to_frequency_bin(*s, *d);
        })
}

/// assuming that the DC sample is at the origin
pub fn subscript_to_freq_bin_f32(dimensions: &[usize], subscript: &[usize], freq_bin: &mut [f32]) {
    dimensions
        .iter()
        .zip(subscript.iter())
        .zip(freq_bin.iter_mut())
        .for_each(|((d, s), val)| {
            *val = index_to_frequency_bin(*s, *d) as f32
        })
}

/// assuming that DC sample is at the origin
pub fn freq_bin_to_subscript(dimensions: &[usize], freq_bin: &[i32], subscript: &mut [usize]) {
    dimensions
        .iter()
        .zip(freq_bin.iter())
        .zip(subscript.iter_mut())
        .for_each(|((d, b), val)| {
            *val = frequency_bin_to_index(*b, *d);
        })
}

/// convert array index to a frequency bin integer, given the array length = dft size
fn index_to_frequency_bin(index: usize, dtf_size: usize) -> i32 {
    // if fft_size is odd, we add 1 to round up
    let half_fft_size = if dtf_size & 1 == 0 {
        dtf_size / 2
    } else {
        (dtf_size / 2) + 1
    };

    if index < half_fft_size {
        index as i32
    } else {
        (index as i32) - (dtf_size as i32)
    }
}

/// convert a frequency bin integer to an array index, given the array length = dft size
fn frequency_bin_to_index(bin: i32, dtf_size: usize) -> usize {
    if bin < 0 {
        (bin + dtf_size as i32) as usize
    } else {
        bin as usize
    }
}

/// down-sample array from the origin. This is useful for re-griding k-space. In this context, the DC
/// sample is assumed to be at index 0
pub fn down_sample(
    x: &ArrayD<Complex32>,
    to_dims: &[usize],
) -> Result<ArrayD<Complex32>, Box<dyn Error>> {
    let from_dims = x.shape();
    if to_dims.len() != from_dims.len() {
        Err(format!(
            "{} dimensions were specified, but array has {} dimensions",
            to_dims.len(),
            from_dims.len()
        ))?
    }

    for (to, from) in to_dims.iter().zip(from_dims.iter()) {
        if to > from {
            Err(format!(
                "target dimensions must be smaller than source dimensions: {} > {}",
                to, from
            ))?;
        }
    }

    let mut result = ArrayD::from_elem(to_dims.f(), Complex32::ZERO);
    let mut idxs = vec![0usize; to_dims.len()];
    let mut freq_bin = vec![0; to_dims.len()];
    let mut sub = vec![0usize; to_dims.len()];

    result.indexed_iter_mut().for_each(|(idx, val)| {
        idxs.iter_mut()
            .zip(idx.as_array_view().iter())
            .for_each(|(a, b)| *a = *b);
        subscript_to_freq_bin(to_dims, &idxs, &mut freq_bin);
        freq_bin_to_subscript(from_dims, &freq_bin, &mut sub);
        *val = *x.get(sub.as_slice()).unwrap_or(&Complex32::ZERO)
    });
    Ok(result)
}

/// up-sample array from the origin. This is useful for re-griding k-space. In this context, the DC
/// sample is assumed to be at index 0
pub fn up_sample(
    x: &ArrayD<Complex32>,
    to_dims: &[usize],
) -> Result<ArrayD<Complex32>, Box<dyn Error>> {
    let from_dims = x.shape();

    if to_dims.len() != from_dims.len() {
        Err(format!(
            "{} dimensions were specified, but array has {} dimensions",
            to_dims.len(),
            from_dims.len()
        ))?
    }

    for (to, from) in to_dims.iter().zip(from_dims.iter()) {
        if to < from {
            Err(format!(
                "target dimensions must be larger than source dimensions: {} < {}",
                to, from
            ))?;
        }
    }

    let mut result = ArrayD::from_elem(to_dims.f(), Complex32::ZERO);

    let mut freq_bin = vec![0; to_dims.len()];
    let mut idxs = vec![0usize; to_dims.len()];
    let mut sub = vec![0usize; to_dims.len()];

    x.indexed_iter().for_each(|(idx, val)| {
        idxs.iter_mut()
            .zip(idx.as_array_view().iter())
            .for_each(|(a, b)| *a = *b);
        subscript_to_freq_bin(from_dims, &idxs, &mut freq_bin);
        freq_bin_to_subscript(to_dims, &freq_bin, &mut sub);

        result[sub.as_slice()] = *val;
    });

    Ok(result)
}

/// re-samples fourier space to the desired dimensions. Target dimensions must be either strictly
/// larger or strictly smaller than the input dimensions
pub fn re_sample(
    x: &ArrayD<Complex32>,
    to_dims: &[usize],
) -> Result<ArrayD<Complex32>, Box<dyn Error>> {
    let from_dims = x.shape();
    if from_dims.len() != to_dims.len() {
        Err(format!(
            "{} dimensions were specified, but array has {} dimensions",
            to_dims.len(),
            from_dims.len()
        ))?
    }

    if to_dims
        .iter()
        .zip(from_dims.iter())
        .any(|(to, from)| *to < *from)
    {
        down_sample(x, to_dims)
    } else {
        up_sample(x, to_dims)
    }
}

/// apply a window function to k-space. Assumes that the DC sample is at the array origin
/// i.e. no fftshift has been applied
pub fn apply_window<W:WindowFunction>(x:&mut ArrayD<Complex32>, window:&W) {

    let n_dims = x.shape().len();
    let dims = x.shape().to_owned();
    let mut freq_bin = vec![0; n_dims];
    let mut idxs = vec![0usize; n_dims];

    x.indexed_iter_mut().for_each(|(idx, val)| {
        idxs.iter_mut()
            .zip(idx.as_array_view().iter())
            .for_each(|(a, b)| *a = *b);
        subscript_to_freq_bin(&dims, &idxs, &mut freq_bin);
        let w = window.evaluate(&freq_bin).to_f32().expect("failed to convert to float");
        *val *= w;
    });
}

/// maps a linear index from one array dimension to another
fn map_linear_index(linear_index: usize, source_shape: &[usize], target_shape: &[usize]) -> usize {
    let mut source_stride = 1;
    let mut target_index = 0;

    for (&source_size, &target_size) in source_shape.iter().rev().zip(target_shape.iter()) {
        if source_size == 0 {
            panic!("dim cannot be 0")
        }
        let source_position = (linear_index / source_stride) % source_size;
        target_index = target_index * target_size + source_position;
        source_stride *= source_size;
    }

    target_index
}

#[cfg(test)]
mod tests {

    use cfl::{ndarray::{ArrayD, ShapeBuilder}, num_complex::Complex32};
    use crate::{rustfft::{apply_window, frequency_bin_to_index, index_to_frequency_bin}, window_functions::{HanningWindow, TukeyWindow, WindowFunction}};
    use super::{fftshift, ifftnc, set_dc_phase, set_dc_to_center, set_dc_to_origin, up_sample};

    #[test]
    fn fft_test() {
        for i in 0..3 {
            let f_in = format!("test_data/k{:02}", i);
            let t_out = format!("test_data/im{:02}", i);
            let k_out = format!("test_data/c{:02}", i);

            // get data ready for centered fft
            let mut x = cfl::to_array(&f_in, true).unwrap();

            set_dc_to_center(&mut x).unwrap();
            set_dc_phase(&mut x, 0.).unwrap();
            //ifftnc(&mut x);

            cfl::from_array(&t_out, &x).unwrap();
        }
    }

    #[test]
    fn re_sample_test() {
        let mut x = cfl::to_array("test_data/k00", true).unwrap();

        set_dc_to_origin(&mut x).unwrap();
        //let mut d = down_sample(&x, &[15,15,15]).unwrap();
        let mut d = up_sample(&x, &[788, 480, 480]).unwrap();
        fftshift(&mut d);

        ifftnc(&mut d);

        cfl::from_array("test_data/ds", &d).unwrap();
    }

    #[test]
    fn freq_bin_test() {
        // odd fft size
        let fft_size = 7;
        let expected_bins = vec![0, 1, 2, 3, -3, -2, -1];
        let idx: Vec<_> = (0..fft_size).collect();
        let bins: Vec<_> = idx
            .iter()
            .map(|x| index_to_frequency_bin(*x, idx.len()))
            .collect();
        let new_idx: Vec<_> = bins
            .iter()
            .map(|x| frequency_bin_to_index(*x, idx.len()))
            .collect();
        assert_eq!(idx, new_idx);
        assert_eq!(bins, expected_bins);

        // even fft size
        let fft_size = 6;
        let expected_bins = vec![0, 1, 2, -3, -2, -1];
        let idx: Vec<_> = (0..fft_size).collect();
        let bins: Vec<_> = idx
            .iter()
            .map(|x| index_to_frequency_bin(*x, idx.len()))
            .collect();
        let new_idx: Vec<_> = bins
            .iter()
            .map(|x| frequency_bin_to_index(*x, idx.len()))
            .collect();
        assert_eq!(idx, new_idx);
        assert_eq!(bins, expected_bins);
    }

    #[test]
    fn apply_window_test() {
        let window = TukeyWindow::new(2.,5.,0.,1.);
        let size:[usize;3] = [100,200,300];
        let mut x = ArrayD::from_elem(size.as_slice().f(), Complex32::ONE);
        apply_window(&mut x, &window);
        cfl::from_array("w_test", &x).unwrap();
        println!("{:?}",x);
    }

    #[test]
    fn hann_window_test() {
        let dims = [128,120,130];
        let window = HanningWindow::new(dims.as_slice());
        let mut x = ArrayD::from_elem(dims.as_slice().f(), Complex32::ONE);
        window.apply(&mut x);
        cfl::from_array("w_test", &x).unwrap();
    }
}




