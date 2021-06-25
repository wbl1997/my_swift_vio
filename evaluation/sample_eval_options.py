
def sample_swiftvio_options():
    # python3.7 will remember insertion order of items, see
    # https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6
    algo_option_templates = {
        'OKVIS': {"algo_code": "OKVIS",
                  "extra_gflags": "--publish_via_ros=false",
                  "displayImages": "false",
                  "monocular_input": 1,
                  "loop_closure_method": 0},
    }

    config_name_to_diffs = {
        ('KSWF', 'OKVIS'): {
            "algo_code": 'HybridFilter',
            "numImuFrames": 5,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('KSWF_n', 'OKVIS'): {
            "algo_code": 'HybridFilter',
            "numImuFrames": 5,
            "monocular_input": 0,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('OKVIS', 'OKVIS'): {
            "sigma_g_c": 12.0e-4 * 4,
            "sigma_a_c": 8.0e-3 * 4,
            "sigma_gw_c": 4.0e-6 * 4,
            "sigma_aw_c": 4.0e-5 * 4,
        },
        ('OKVIS_n', 'OKVIS'): {
            "monocular_input": 0,
            # We override P_C0C_Q_C0C as it conflicts with okvis estimator.
            "extrinsic_opt_mode_other_camera": "P_BC_Q_BC",
            "sigma_g_c": 12.0e-4 * 4,
            "sigma_a_c": 8.0e-3 * 4,
            "sigma_gw_c": 4.0e-6 * 4,
            "sigma_aw_c": 4.0e-5 * 4,
        },
        ('KSF', 'OKVIS'): {
            "algo_code": 'MSCKF',
            "numImuFrames": 5,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('KSF_n', 'OKVIS'): {
            "algo_code": 'MSCKF',
            "numImuFrames": 5,
            "monocular_input": 0,
            "sigma_absolute_translation": 0.02,
            "sigma_absolute_orientation": 0.01,
            "model_type": "BG_BA",
            'projection_opt_mode': 'FIXED',
            "extrinsic_opt_mode_main_camera": "P_CB",
            "extrinsic_opt_mode_other_camera": "P_C0C_Q_C0C",
        },
        ('isam2-fls', 'OKVIS'): {
            "algo_code": "SlidingWindowSmoother",
            "extra_gflags": "--publish_via_ros=false",
        },
        ('ri-fls', 'OKVIS'): {
            "algo_code": "RiSlidingWindowSmoother",
            "extra_gflags": "--publish_via_ros=false --rifls_lock_jacobian=true",
        },
        ('ri-fls-exact', 'OKVIS'): {
            "algo_code": "RiSlidingWindowSmoother",
            "extra_gflags": "--publish_via_ros=false --rifls_lock_jacobian=false",
        },
    }
    return algo_option_templates, config_name_to_diffs
