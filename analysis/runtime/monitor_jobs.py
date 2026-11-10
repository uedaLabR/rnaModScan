"""Sample existing Linux process groups without changing the running jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return dict(command=command, returncode=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, error=str(error))


def numeric_fields(path):
    values = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(':')
        fields = value.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    return values


def read_process(path, groups):
    # comm can contain spaces or parentheses; fields after its final ')' are fixed.
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    pgid = int(fields[2])
    if pgid not in groups or fields[0] == 'Z':
        return None
    ticks = os.sysconf('SC_CLK_TCK')
    status = numeric_fields(path / 'status')
    try:
        io = numeric_fields(path / 'io')
    except OSError:
        io = None
    return dict(pid=int(path.name), pgid=pgid, state=fields[0],
                start_ticks=int(fields[19]),
                user_seconds=int(fields[11]) / ticks,
                system_seconds=int(fields[12]) / ticks,
                rss_kib=status.get('VmRSS'), individual_peak_rss_kib=status.get('VmHWM'),
                threads=status.get('Threads'), io=io,
                command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip())


def read_process_groups(groups):
    processes = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            process = read_process(path, groups)
            if process:
                processes.append(process)
        except (OSError, ValueError, IndexError):
            continue  # A process may exit during a sample.
    return processes


def gpu_snapshot():
    return dict(
        devices=command_output(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader,nounits']),
        processes=command_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                  '--format=csv,noheader,nounits']))


def monitor(jobs, output, interval):
    output.mkdir(parents=True, exist_ok=False)
    processes = read_process_groups(set(jobs.values()))
    if not all(any(p['pgid'] == group for p in processes) for group in jobs.values()):
        raise RuntimeError('A requested process group is absent; check host /proc visibility and job IDs.')
    metadata = dict(started_utc=utc_now(), monitor_pid=os.getpid(), jobs=jobs,
                    interval_seconds=interval, hostname=platform.node(), platform=platform.platform(),
                    cpu=command_output(['lscpu']), memory=Path('/proc/meminfo').read_text(),
                    gpu=command_output(['nvidia-smi']),
                    limitations=[
                        'Monitoring starts mid-run; earlier aggregate RAM/GPU peaks are unknown.',
                        'Sampled peaks are lower bounds, not exact peaks; short-lived children may be missed.',
                        'RSS sum can double-count shared pages; VmHWM is per process, not a group peak.',
                        'Process CPU/IO counters are lifetime values, not interval increments.',
                        'GPU utilization is device-wide; associate per-process memory by sampled PID.',
                        'Concurrent jobs share CPU/storage; this is an observational run, not an isolated benchmark.',
                        'Job completion and success must be determined from application logs, not process disappearance.'])
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    previous = {}
    previous_time = time.monotonic()
    active = dict(jobs)
    with (output / 'samples.jsonl').open('x', buffering=1) as handle:
        while active:
            started = time.monotonic()
            processes = read_process_groups(set(active.values()))
            groups = {}
            elapsed = started - previous_time
            current = {}
            for name, pgid in active.items():
                members = [p for p in processes if p['pgid'] == pgid]
                delta = 0.0
                for p in members:
                    key = (p['pid'], p['start_ticks'])
                    cpu = p['user_seconds'] + p['system_seconds']
                    if key in previous:
                        delta += max(0.0, cpu - previous[key])
                    current[key] = cpu
                groups[name] = dict(pgid=pgid, process_count=len(members),
                                    rss_sum_kib=sum(p['rss_kib'] or 0 for p in members),
                                    cpu_percent=100 * delta / elapsed if previous and elapsed > 0 else None,
                                    processes=members)
            sample = dict(utc=utc_now(), interval_seconds=elapsed, jobs=groups, gpu=gpu_snapshot())
            handle.write(json.dumps(sample) + '\n')
            previous, previous_time = current, started
            active = {name: pgid for name, pgid in active.items() if groups[name]['process_count']}
            if active:
                time.sleep(max(0.0, interval - (time.monotonic() - started)))
    (output / 'finished.json').write_text(json.dumps(dict(finished_utc=utc_now(), reason='All watched groups exited.')) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', action='append', required=True, help='NAME:PGID; repeat for each job')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least one second')
    jobs = {}
    for value in args.job:
        name, group = value.rsplit(':', 1)
        if name in jobs:
            parser.error('Job names must be unique')
        jobs[name] = int(group)
    monitor(jobs, args.output, args.interval)


if __name__ == '__main__':
    main()
