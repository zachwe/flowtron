import argparse
import signal
import subprocess

from .train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()

    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = "29500"
    current_env["WORLD_SIZE"] = n_gpus

    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None
    subprocs = []
    for r in range(n_gpus):
        current_env["RANK"] = r
        current_env["LOCAL_RANK"] = r
        cmd = ["train.py"]
        def sigkill_handler(signum, frame):
            for process in subprocs:
                print(f"Killing subprocess {process.pid}")
                try:
                    process.kill()
                except Exception:
                    pass
            if last_return_code is not None:
                raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
            if signum in sig_names:
                print(f"Main process received {sig_names[signum]}, exiting")
            sys.exit(1)

        # pass SIGINT/SIGTERM to children if the parent is being terminated
        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

        stdout_handle = None if not subprocess_file_handles else subprocess_file_handles[r][0]
        stderr_handle = None if not subprocess_file_handles else subprocess_file_handles[r][1]
        cmd.extend(args.training_script_args)

        process = subprocess.Popen(cmd, env=current_env, stdout=stdout_handle, stderr=stderr_handle)
        subprocs.append(process)

    try:
        alive_processes = set(subprocs)
        while len(alive_processes):
            finished_processes = []
            for process in alive_processes:
                if process.poll() is None:
                    # the process is still running
                    continue
                else:
                    if process.returncode != 0:
                        last_return_code = process.returncode  # for sigkill_handler
                        sigkill_handler(signal.SIGTERM, None)  # not coming back
                    else:
                        # exited cleanly
                        finished_processes.append(process)
            alive_processes = set(alive_processes) - set(finished_processes)

            time.sleep(1)
