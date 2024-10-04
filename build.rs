fn main() {

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();

    let c_file = "src/fftwf_exec.c";
    let obj = "fftwf_exec";

    if target_os == "linux" {
        // lets assume that clang can find these libraries on linux :p
        cc::Build::new()
        .file(c_file)
        .compile(obj);

        println!("cargo:rustc-link-lib=fftw3f");
        println!("cargo:rustc-link-lib=fftw3f_threads");
        println!("cargo:rustc-link-lib=gomp");  // OpenMP support

    }else if target_os == "macos" {
        // hard-coded include paths for mac
        cc::Build::new()
        .file(c_file)
        .include("/usr/local/opt/fftw/include")
        .include("/usr/local/opt/libomp/include")
        .compile(obj);

        println!("cargo:rustc-link-search=native=/usr/local/opt/fftw/lib");
        println!("cargo:rustc-link-lib=static=fftw3f");
        println!("cargo:rustc-link-lib=static=fftw3f_threads");
        println!("cargo:rustc-link-search=native=/usr/local/opt/libomp/lib");
        println!("cargo:rustc-link-lib=static=omp")
    }

}
