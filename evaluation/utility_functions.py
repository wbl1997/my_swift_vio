import subprocess

def subprocess_cmd_check(command):
    """Run command in a subprocess shell and wait until completion by default"""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as err:
        print('Error code {} in running command {} with output:\n{}\n'.format(err.returncode, command, err))
        return err.returncode, err.output
    except Exception as err:
        print("Unexpected error:{}".format(err))
        return 1, "Unexpected error"
    return 0, output


def subprocess_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    # wait for the process to terminate
    out, err = process.communicate()
    if process.returncode != 0:
        print('cmd {}\nreturn code:{} stdout:{}\nstderr:{}\n'.format(cmd, process.returncode, out.strip(), err.strip()))
    return process.returncode, err.strip()


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
