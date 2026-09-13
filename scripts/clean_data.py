"""Report what a move would carry, and optionally drop what can be rebuilt.

Moving this folder means answering one question per directory: is this source, is
it the state the models were trained on, or is it something the code can produce
again? Only the middle category is worth its size.

There is no way to answer that from the code alone. A dataset is 1.5 GB and takes
two minutes to rebuild; the candle database is 1.4 GB and takes hours. Sizing them
the same way is how a "small" copy turns out to be six gigabytes.

    python -m scripts.clean_data              # report only, changes nothing
    python -m scripts.clean_data --drop-safe  # delete the rebuildable parts
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _size(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob('*'):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _human(count):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if count < 1024 or unit == 'GB':
            return '%.1f %s' % (count, unit) if unit != 'B' else '%d B' % count
        count /= 1024.0
    return '%.1f GB' % count


# name, path, rebuildable, what it is
ITEMS = [
    ('代码与测试', 'app', False, '服务本体，必须搬'),
    ('代码与测试', 'tests', False, '906 个测试，必须搬'),
    ('代码与测试', 'scripts', False, '运维脚本，必须搬'),
    ('代码与测试', 'web', False, '前端资源，必须搬'),
    ('代码与测试', 'docs', False, '审计与文档，建议搬'),
    ('配置', '.env.example', False, '可移植模板，必须搬'),
    ('配置', '.env', False, '本机配置，按需搬（含代理设置）'),
    ('配置', 'requirements.txt', False, '依赖清单，必须搬'),
    ('配置', 'requirements-models.txt', False, '模型依赖，必须搬'),
    ('配置', 'start_paper.bat', False, '启动脚本，必须搬'),
    ('配置', 'stop_paper.bat', False, '停止脚本，必须搬'),
    ('模型', 'data/models', False, '晋级模型（小，是训练成果，必须搬）'),
    ('模型', 'data/research_v3/candidates', False, '线上候选，必须搬'),
    ('模型', 'data/research_v3/candidates_mainstream', False, '分层候选，建议搬'),
    ('模型', 'data/research_v3/candidates_speculative', False, '分层候选，建议搬'),
    ('数据库', 'data/research.sqlite3', False,
     'K 线/资金费/持仓量历史。重建要数小时，建议搬'),
    ('可再生', 'data/research_v3/training_dataset_mainstream.jsonl', True,
     '训练数据集，约 2 分钟可重建（并行构建）'),
    ('可再生', 'data/research_v3/training_dataset_speculative.jsonl', True,
     '训练数据集，约 2 分钟可重建'),
    ('可再生', 'data/research_v3/training_dataset.jsonl.profile.json', True, '数据集摘要缓存'),
    ('可再生', 'data/research_v3/training_dataset_mainstream.jsonl.profile.json', True, '数据集摘要缓存'),
    ('可再生', 'data/pretrained', True, 'Chronos 权重，可用 scripts/download_models.py 重新下载'),
    ('可再生', 'data/archive_cache', True, 'Binance 归档 zip 缓存，回补时自动重建'),
    ('可再生', 'data/wheels', True, 'pip wheel 缓存'),
    ('可再生', 'logs', True, '运行日志'),
    ('可再生', 'data/ui_checks', True, '界面检查截图'),
    ('可再生', 'data/backfill_state', True, '回补进度，重建时会重新扫描'),
    ('可再生', 'data/backfill_metrics.log', True, '回补日志'),
    ('可再生', 'data/service.log', True, '服务日志'),
    ('可再生', 'data/paper_run.log', True, '运行日志'),
    ('可再生', 'catboost_info', True, 'CatBoost 临时输出'),
    ('可再生', '.pytest_cache', True, 'pytest 缓存'),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--drop-safe', action='store_true',
                        help='delete the rebuildable items instead of only listing them')
    parser.add_argument('--root', default='.', help='the folder to inspect')
    args = parser.parse_args(argv)

    root = Path(args.root)
    keep_total = drop_total = 0
    print('%-10s %-52s %-10s %s' % ('类别', '路径', '大小', '说明'))
    print('-' * 118)
    for group, rel, rebuildable, note in ITEMS:
        path = root / rel
        size = _size(path)
        if size == 0:
            continue
        if rebuildable:
            drop_total += size
        else:
            keep_total += size
        print('%-10s %-52s %-10s %s' % (group, rel, _human(size), note))

    print('-' * 118)
    print('必须搬/建议搬 : %s' % _human(keep_total))
    print('可删除再生   : %s' % _human(drop_total))
    print('合计         : %s' % _human(keep_total + drop_total))
    print()
    if not args.drop_safe:
        print('未做任何改动。加 --drop-safe 删除"可再生"那一类。')
        print('删除后首次训练会重建数据集，首次回补会重建归档缓存。')
        return 0

    for group, rel, rebuildable, _note in ITEMS:
        if not rebuildable:
            continue
        path = root / rel
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass
        print('已删除 %s' % rel)
    print()
    print('完成后请运行：python -m pytest tests -q  确认仍然 906 通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
