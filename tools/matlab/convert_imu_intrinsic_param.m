function kalibr_estimates = convert_imu_intrinsic_param(msckf_estimates, voicebox_path)
% OBSOLETE: this function performs poor in decomposing T_a = L * R(r,p,y)

addpath(voicebox_path);
% In msckf2 implementation
% w_m = T_g * w_b + T_s * a_b + b_w + n_w
% a_m = T_a * a_b + b_a + n_a
% where body frame {b} is aligned with the camera frame in orientation,
% but its origin is at the accelerometer triad intersection,
% T_g, T_s, and T_a are fully populated

% In ICRA 16 Extending Kalibr, ignoring the lever arm between
% accelerometers
% w_m = T_g * w_b + T_s * a_b + b_w + n_w
% a_m = T_a * a_b + b_a + n_a
% where b is aligned with the x-axis of the accelerometer triad, and has an
% origin at the accelerometer triad intersection
% T_g, T_s is fully populated, T_a is a lower triangular matrix

% below denotes b of msckf by bp(pseudo), and b of kalibr by b
% locate the entries
T_gp =  msckf_estimates(:, Msckf2Constants.T_g);
T_sp =  msckf_estimates(:, Msckf2Constants.T_s);
T_ap =  msckf_estimates(:, Msckf2Constants.T_a);
p_BpC = msckf_estimates(:, Msckf2Constants.p_BC);

q_BBp = zeros(size(msckf_estimates, 1), 4);
R_BpB = cell(size(msckf_estimates, 1));

for i=1:size(T_ap, 1)
    T_ap_mat_t = reshape(T_ap(i, :), 3, 3);
    [Q, R] = qr(T_ap_mat_t);
    cond = eye(3);
    for j=1:3
        if Q(j, j) < 0
            cond(j, j) = -1;
        end
    end
    Q = Q * cond;
    R = cond * R;
    q_BBp(i, :) = rotro2qr(Q'); % w x y z for sanity check
    
    R_BpB{i} = Q;
    if ~isempty(find(abs(q_BBp(i, 2:4)) > 0.5, 1))
        q_BBp(i, :) = [1, 0, 0, 0];
        R_BpB{i} = eye(3);
    end
end

% output
T_g = T_gp;
T_s = T_sp;
T_a = T_ap;
p_BC = p_BpC;

for i=1:size(T_ap, 1)
    p_BC(i, :) = p_BpC(i, :) * R_BpB{i};
    T_a(i, :) = multiply_rot(T_ap(i, :), R_BpB{i});
    T_s(i, :) = multiply_rot(T_sp(i, :), R_BpB{i});
    T_g(i, :) = multiply_rot(T_gp(i, :), R_BpB{i});
end
kalibr_estimates = [msckf_estimates, q_BBp];
kalibr_estimates(:, Msckf2Constants.p_BC) = p_BC;
kalibr_estimates(:, Msckf2Constants.T_g) = T_g;
kalibr_estimates(:, Msckf2Constants.T_s) = T_s;
kalibr_estimates(:, Msckf2Constants.T_a) = T_a;
end

function vec9xrot = multiply_rot(vec1x9, rot3x3)
  mat_t = reshape(vec1x9, 3, 3);
  vec9xrot = reshape(rot3x3' * mat_t, 1, 9);
end