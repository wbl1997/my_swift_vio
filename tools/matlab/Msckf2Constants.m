classdef Msckf2Constants
    % Each line of Msckf2 output csv file except for the header line is an
    % array of values with indices listed below
    
  properties (Constant)
  r = 3:5;
  q = 6:9;
  v = 10:12;
  b_g = 13:15;
  b_a = 16:18;
  T_g = 19:27;
  T_g_diag = [19, 23, 27];
  T_s = 28:36;
  T_s_diag = [28, 32, 36];
  T_a = 37:45;
  T_a_diag = [37, 41, 45];
  p_BC = 46:48;
  fxy_cxy = 49:52;
  k1_k2 = 53:54;
  p1_p2 = 55:56;
  td = 57;
  tr = 58;
  
  r_std = 59:61;
  q_std = 62:64;
  v_std = 65:67;
  b_g_std = 68:70;
  b_a_std = 71:73;
  T_g_std = 74:82;
  T_s_std = 83:91;
  T_a_std = 92:100;
  p_BC_std = 101:103;
  fxy_cxy_std = 104:107;
  k1_k2_std = 108:109;
  p1_p2_std = 110:111;
  td_std = 112;
  tr_std = 113;
  end
end