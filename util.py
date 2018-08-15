def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]