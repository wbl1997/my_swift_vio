import subprocess

def subprocess_cmd(command, block=True):
    """Run command in a subprocess shell and wait until completion by default"""
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    # TODO(jhuai): This risks deadlock caused by wait(), see
    # https://docs.python.org/3/library/subprocess.html
    if block:
        process.wait()
    return process.returncode, proc_stdout

def de_underscore(strin):
    return strin.replace('_', '-')

def load_rel_error_stats(rel_error_stat_txt):
    algo_rel_trans_rot = {}
    with open(rel_error_stat_txt, 'r') as stream:
        for line in stream:
            if "Translation" in line:
                continue
            if line.strip('\n'):
                stats = [segment.strip() for segment in line.split('&')]
                algo_rel_trans_rot[stats[0]] = [float(stats[1]), float(stats[2])]
    return algo_rel_trans_rot

def get_arg_value_from_gflags(cmd_gflags, key_str):
    index = cmd_gflags.find(key_str)
    if index == -1:
        return None
    else:
        # 1 for "="
        val_start_index = index + len(key_str) + 1
        next_arg_index = cmd_gflags.find('--', val_start_index)
        if next_arg_index == -1:
            return cmd_gflags[val_start_index:]
        else:
            return cmd_gflags[val_start_index:next_arg_index]
