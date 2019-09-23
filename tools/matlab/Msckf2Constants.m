classdef Msckf2Constants
    % Each line of Msckf2 output csv file except for the header line is an
    % array of values with indices listed below
    
    properties (Constant)
        r = 3:5;
        q = 6:9;
        v = 10:12;
        b_g = 13:15;
        b_a = 16:18;
    end
    properties       
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
    methods
      function obj = Msckf2Constants(misalignment_dim, extrinsic_dim, ...
              project_intrinsic_dim, distort_intrinsic_dim)
        obj.T_g = 19:27;
        obj.T_g_diag = [19, 23, 27];
        obj.T_s = 28:36;
        obj.T_s_diag = [28, 32, 36];
        obj.T_a = 37:45;
        obj.T_a_diag = [37, 41, 45];
        param_index = 46;
        
        obj.p_BC = param_index + (0:2);
        param_index = param_index + extrinsic_dim;
        switch project_intrinsic_dim
            case 1
                obj.fxy_cxy = param_index + [0, 0, 0, 0];
            case 3
                obj.fxy_cxy = param_index + [0, 0, 1, 2];
            otherwise
                obj.fxy_cxy = param_index + (0:3);
        end
        param_index = param_index + project_intrinsic_dim;
        
        switch distort_intrinsic_dim
            case 1
                obj.k1_k2 = param_index + [0, 0];
                obj.p1_p2 = param_index + [0, 0];
            otherwise
                obj.k1_k2 = param_index + (0:1);
                obj.p1_p2 = param_index + (2:3);
        end
        param_index = param_index + distort_intrinsic_dim;
        obj.td = param_index;
        obj.tr = param_index + 1;
        param_index = param_index + 2;
        
        obj.r_std = param_index + (0:2);
        obj.q_std = param_index + (3:5);
        obj.v_std = param_index + (6:8);
        obj.b_g_std = param_index + (9:11);
        obj.b_a_std = param_index + (12:14);
        param_index = param_index + 15;
        
        obj.T_g_std = param_index + (0:8);
        obj.T_s_std = param_index + (9:17);
        obj.T_a_std = param_index + (18:26);
        
        param_index = param_index + misalignment_dim;
        
        obj.p_BC_std = param_index + (0:2);
        param_index = param_index + extrinsic_dim;
        switch project_intrinsic_dim
            case 1
                obj.fxy_cxy_std = param_index + [0, 0, 0, 0];
            case 3
                obj.fxy_cxy_std = param_index + [0, 0, 1, 2];
            otherwise
                obj.fxy_cxy_std = param_index + (0:project_intrinsic_dim-1);
        end
        param_index = param_index + project_intrinsic_dim;
        switch distort_intrinsic_dim
            case 1
                obj.k1_k2_std = param_index + [0, 0];
                obj.p1_p2_std = param_index + [0, 0];
            otherwise
                obj.k1_k2_std = param_index + (0:1);
                obj.p1_p2_std = param_index + (2:3);
        end
        param_index = param_index + distort_intrinsic_dim;
        obj.td_std = param_index;
        obj.tr_std = param_index + 1;
      end
   end
end