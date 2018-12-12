% https://nghiaho.com/?page_id=846
function [eul_xyz] = rotmtx_to_eul_xyz(rot_mtx)
    tx = atan2(rot_mtx(3, 2), rot_mtx(3, 3));
    ty = atan2(-rot_mtx(3, 1), sqrt(rot_mtx(3, 2)^2 + rot_mtx(3, 3)^2));
    tz = atan2(rot_mtx(2, 1), rot_mtx(1, 1));
    eul_xyz = [tx, ty, tz];
end

