"""Export measured CPU inference time and historical GPU timing without mixing scopes."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.io.excel import write_workbook, verify_workbook
from utils.references.catalog import sha256


def summarize(benchmark, output):
    benchmark, output = Path(benchmark).resolve(), Path(output).resolve()
    run = json.loads((benchmark / 'run.json').read_text())
    if run['status'] != 'complete':
        raise ValueError('Wait for the complete CPU inference and score comparison')
    output.mkdir(parents=True, exist_ok=False)
    sites = run['sites']
    times = [
        ['NN inference', 'GPU', 'historical measurement', run['gpu_inference_seconds'], sites],
        ['NN inference', 'CPU', 'new measurement', run['measured_cpu_inference_seconds'], sites],
        ['Full workflow (four targets)', 'GPU', 'historical measurement', run['full_gpu_workflow_seconds'], None],
        ['Full workflow (four targets)', 'CPU', 'estimate: historical CPU preparation/filter + new inference',
         run['estimated_cpu_full_workflow_seconds'], None],
    ]
    timing_headers = ['scope', 'device', 'measurement', 'wall_seconds', 'sites', 'wall_minutes', 'sites_per_second']
    timing_rows = [row + [row[3] / 60, row[4] / row[3] if row[4] is not None else None] for row in times]
    comparison = run['score_comparison']
    agreement = [
        ['evaluated_sites', comparison['sites'], 'sites'],
        ['argmax_agreement', comparison['argmax_agreement'], 'fraction'],
        ['class_changes', comparison['class_changes'], 'sites'],
        ['max_absolute_probability_difference', comparison['max_absolute_probability_difference'], 'probability'],
        ['mean_absolute_probability_difference', comparison['mean_absolute_probability_difference'], 'probability'],
    ]
    for name, counts in comparison['class_counts'].items():
        agreement.extend([[name + '_GPU', counts['gpu'], 'sites'], [name + '_CPU', counts['cpu'], 'sites']])
    for row in agreement:
        if row[2] == 'probability':
            display = f'{row[1]:.6e}'
        elif row[2] == 'fraction':
            display = f'{row[1] * 100:.6f}%'
        else:
            display = str(row[1])
        row.append(display)
    resources = run['resources']
    hardware = (benchmark / 'cpu_hardware.txt').read_text().splitlines()
    cpu_model = next((line.split(':', 1)[1].strip() for line in hardware if line.startswith('Model name:')),
                     'See benchmark cpu_hardware.txt')
    environment = [
        ['hostname', run['hostname']], ['platform', run['platform']],
        ['CPU', cpu_model], ['Available logical CPUs (affinity)', len(run['cpu_affinity'])],
        ['TF intra-op threads', run['tensorflow_intra_threads']], ['TF inter-op threads', 1],
        ['TF thread setting scope', 'TensorFlow setting; not a whole-process physical-core limit'],
        ['batch_sites', run['batch_sites']], ['GPU visible to CPU inference', False],
        ['TensorFlow version', comparison['runtime']['tensorflow_version']],
        ['CPU user_seconds', float(resources['user_seconds'])],
        ['CPU system_seconds', float(resources['system_seconds'])],
        ['CPU peak_RSS_KiB', int(resources['max_rss_kib'])],
        ['CPU peak_RSS_GiB', int(resources['max_rss_kib']) / 1024**2],
        ['CPU average process usage', resources['cpu_percent']],
        ['CPU/GPU inference wall-time ratio', run['cpu_over_historical_gpu_time_ratio']],
        ['Prepared input manifest SHA256', run['input_run_sha256']],
        ['Model SHA256', run['model_sha256']],
        ['Timing scope', run['timing_scope']],
        ['Comparison limitation', 'Same host, inputs, weights, batch and TF setting; separate historical runs, not an isolated concurrent benchmark'],
    ]
    sheets = [dict(name='Timing', headers=timing_headers, rows=timing_rows),
              dict(name='Prediction_agreement', headers=['metric', 'value', 'unit', 'display'], rows=agreement),
              dict(name='Environment', headers=['item', 'value'], rows=environment)]
    workbook = output / 'HeLa_CPU_GPU_runtime_20261109.xlsx'
    write_workbook(workbook, sheets, float_format='0.0000')
    restored = verify_workbook(workbook)
    for sheet in sheets:
        if restored[sheet['name']] != [sheet['headers'], *sheet['rows']]:
            raise ValueError('Workbook round-trip mismatch')
    with (output / 'timing.tsv').open('x', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(timing_headers)
        writer.writerows(timing_rows)
    (output / 'score_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    report = dict(status='complete', source=str(benchmark),
                  source_run_sha256=sha256(benchmark / 'run.json'), workbook=workbook.name,
                  workbook_sha256=sha256(workbook), all_cells_verified=True,
                  actual_CPU_full_workflow_measured=False)
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'README.txt').write_text(
        f"HeLa {sites:,}候補、同一の保存済みNN入力とモデル。batch16。\n"
        f"CPU推論実測：{run['measured_cpu_inference_seconds']/60:.2f}分。"
        f"過去GPU推論：{run['gpu_inference_seconds']/60:.2f}分。\n"
        f"CPU/GPU所要時間比：{run['cpu_over_historical_gpu_time_ratio']:.3f}。\n"
        f"CPU推論の最大RSS：{int(resources['max_rss_kib'])/1024**2:.2f}GiB（GNUtime）。\n"
        f"クラス一致率：{comparison['argmax_agreement']*100:.6f}%、判定差{comparison['class_changes']}部位。\n"
        f"確率差の最大：{comparison['max_absolute_probability_difference']:.6e}、"
        f"平均：{comparison['mean_absolute_probability_difference']:.6e}。\n"
        "時間はモデル/Parquet読み込み・復元・NN・score保存を含み、計算kernelのみではありません。\n"
        "CPU4設定はTF intra-op4/inter-op1です。Arrow等の補助threadを含むプロセス全体のコア上限ではありません。\n"
        "全体CPU時間は、以前の準備/抽出/注釈/フィルタ時間と今回のCPU推論を合算した推定です。\n"
        "BAMからのCPU全工程を再実行した値ではありません。GPUは別日時の既存実測です。\n"
        "元のBAM、学習済み重み、抽出データ、GPU出力、論文原稿は変更していません。\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'results/benchmark_HeLa_CPU_20261109')
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/data/cpu_benchmark_20261109')
    args = parser.parse_args()
    summarize(args.benchmark, args.output)
