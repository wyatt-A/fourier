use std::ffi::c_char;
use cfl::num_complex::Complex32;

extern "C" {
    pub fn execute(
        rank: ::std::os::raw::c_int,
        n: *const ::std::os::raw::c_int,
        sign: ::std::os::raw::c_int,
        wisdom_string_in: *const ::std::os::raw::c_char,
        wisdom_string_out: *mut *mut c_char,
        in_: *mut Complex32,
        out: *mut Complex32,
    ) -> ::std::os::raw::c_int;
    pub fn free_c_string(s:*mut ::std::os::raw::c_char);
}