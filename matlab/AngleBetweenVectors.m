function [angle] = AngleBetweenVectors(u,v)
    u_unit = u / norm(u);
    v_unit = v / norm(v);
    angle = atan2(norm(cross(u_unit,v_unit)), dot(u_unit,v_unit));
end

