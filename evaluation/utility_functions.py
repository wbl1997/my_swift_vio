import subprocess

def subprocess_cmd(command, out_stream=subprocess.STDOUT, err_stream=subprocess.STDOUT):
    """
    Run command in a subprocess shell and wait until completion by default.
    This is a reference implementation for debugging.
    """
    try:
        rc = subprocess.call(command, stdout=out_stream, stderr=err_stream, shell=True, close_fds=True)
    except Exception as err:
        print("Unexpected error:{}".format(err))
        return 1, "Unexpected error"
    return rc, ""


def subprocess_cmd_unsafe(cmd, out_stream=subprocess.PIPE, err_stream=subprocess.PIPE):
    """
    Run command in a subprocess shell and wait until completion.
    If subprocess cmd is going to produce much output, then do not use
    subprocess.PIPE for out_stream or err_stream which may cause deadlock.
    Even so, Popen occasionally deadlocks.

    :param cmd: string of multiple semi-colon separated bash commands
    :param out_stream:
    :param err_stream:
    :return:
    """

    process = subprocess.Popen(cmd, shell=True,
                               stdout=out_stream,
                               stderr=err_stream,
                               close_fds=True)
    # wait for the process to terminate
    out, err = process.communicate()
    if process.returncode != 0:
        print('cmd {}\nreturn code:{} stdout:{}\nstderr:{}\n'.format(
            cmd, process.returncode, out, err))
    return process.returncode, err


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
