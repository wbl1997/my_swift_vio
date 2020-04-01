def create_algo_config(val_list):
    d = {"algo_code": val_list[0],
         "extra_gflags": val_list[1],
         "numKeyframes": val_list[2],
         "numImuFrames": val_list[3],
         "monocular_input": 1
         }
    if len(val_list) >= 5:
        d["monocular_input"] = val_list[4]
    return d
