use cfl::{ndarray::{Array3, ArrayD, ShapeBuilder}, num_complex::{Complex, Complex32}};
use crate::rustfft::{fftn, subscript_to_freq_bin_f32};

/// returns the laplacian kernel in 3-D in fourier space.
/// This is useful for computing the laplacian of arrays
/// for phase unwrapping applications
pub fn laplace_kernel_3d(dims:[usize;3]) -> ArrayD<Complex32> {
    

    for d in dims {
        assert!(d > 8,"kernel size too small to avoid boundary effects. Must be larger than 8");
    }

    let mut array = ArrayD::from_elem(dims.as_slice().f(), Complex::ZERO);

    array[[0,0,0]] = Complex32::new(6.0,0.);
    array[[1,0,0]] = Complex32::new(-1.0,0.);
    array[[dims[0]-1,0,0]] = Complex32::new(-1.0,0.);
    array[[0,1,0]] = Complex32::new(-1.0,0.);
    array[[0,dims[1]-1,0]] = Complex32::new(-1.0,0.);
    array[[0,0,1]] = Complex32::new(-1.0,0.);
    array[[0,0,dims[2]-1]] = Complex32::new(-1.0,0.);

    fftn(&mut array);
    array
}

/// returns the unit dipole kernel in the frequency domain
pub fn dipole_kernel_3d(dims:[usize;3],field_direction:[f32;3],voxel_size:[f32;3]) -> Array3<Complex32> {

    let mut result = Array3::<Complex32>::zeros(dims.f());
    let mut freq_bin = vec![0.; 3];

    result.indexed_iter_mut().for_each(|(idx, val)| {
        subscript_to_freq_bin_f32(&dims, &[idx.0,idx.1,idx.2], &mut freq_bin);
        
        freq_bin.iter_mut().zip(dims).zip(voxel_size).for_each(|((val,dim),vox_size)|{
            *val /= dim as f32 * vox_size
        });

        // unit dipole formulation in frequency domain
        //D = 1/3-  ( X*B0_dir(1) + Y*B0_dir(2) + Z*B0_dir(3) ).^2./(X.^2+Y.^2+Z.^2);
        let num = freq_bin.iter().zip(field_direction).map(|(&x,f)| x * f).sum::<f32>().powi(2);
        let denom = freq_bin.iter().map(|&x| x * x ).sum::<f32>();
        let x = 1./3. - ( num / denom);

        if !x.is_nan() {
            val.re = x;
        }

    });

    return result

}


/*
    [Y,X,Z]=meshgrid(-matrix_size(2)/2:(matrix_size(2)/2-1),...
        -matrix_size(1)/2:(matrix_size(1)/2-1),...
        -matrix_size(3)/2:(matrix_size(3)/2-1));
    
    X = X/(matrix_size(1)*voxel_size(1));
    Y = Y/(matrix_size(2)*voxel_size(2));
    Z = Z/(matrix_size(3)*voxel_size(3));
    
    D = 1/3-  ( X*B0_dir(1) + Y*B0_dir(2) + Z*B0_dir(3) ).^2./(X.^2+Y.^2+Z.^2);
    D(isnan(D)) = 0;
    D = fftshift(D);
*/



#[cfg(test)]
mod tests {
    use super::laplace_kernel_3d;

    #[test]
    fn laplace_kern_test() {
        let k = laplace_kernel_3d([9,9,9]);
        println!("{:?}",k);
    }

}