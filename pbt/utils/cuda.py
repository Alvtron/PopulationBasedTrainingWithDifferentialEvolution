import subprocess

def get_gpu_memory_stats():
    """Get the current gpu memory usage.

    Returns
    -------
    dict\n
        Keys are device ids as integers.\n
        Values are memory usage as a tuple of (memory used, memory total) as integers in MB.
    """
    encoding = 'utf-8'
    memory_used_query = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding=encoding)
    memory_total_query = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.total',
            '--format=csv,nounits,noheader'
        ], encoding=encoding)
    # Convert lines into a dictionary
    gpu_memory_used = [int(x) for x in memory_used_query.strip().split('\n')]
    gpu_memory_total = [int(x) for x in memory_total_query.strip().split('\n')]
    gpu_memory_map = dict(enumerate(zip(gpu_memory_used, gpu_memory_total)))
    return gpu_memory_map