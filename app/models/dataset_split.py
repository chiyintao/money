"""Time-grouped partitions with purging of overlapping forward labels."""
from ..features.quality import validate_dataset


def chronological_split(rows, train_ratio=.7, valid_ratio=.15, minimum=5):
    quality = validate_dataset(rows)
    if not quality['ready']:
        return [], [], [], {'ready': False, 'reason': 'invalid_dataset', 'quality': quality}
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        return [], [], [], {'ready': False, 'reason': 'invalid_split_ratios'}
    ordered = sorted(rows, key=lambda row: (row['timestamp'], row['symbol']))
    times = sorted({row['timestamp'] for row in ordered})
    cut = int(len(times) * train_ratio)
    valid_cut = cut + int(len(times) * valid_ratio)
    if not 0 < cut < valid_cut < len(times):
        return [], [], [], {'ready': False, 'reason': 'insufficient_time_groups'}
    valid_start, test_start = times[cut], times[valid_cut]
    train = [r for r in ordered if r['timestamp'] < valid_start and r.get('label_end_time', r['timestamp']) < valid_start]
    valid = [r for r in ordered if valid_start <= r['timestamp'] < test_start and r.get('label_end_time', r['timestamp']) < test_start]
    test = [r for r in ordered if r['timestamp'] >= test_start]
    ready = min(len(train), len(valid), len(test)) >= minimum
    metadata = {'ready': ready, 'reason': None if ready else 'split_below_minimum',
                'train': len(train), 'validation': len(valid), 'test': len(test),
                'validation_start': valid_start, 'test_start': test_start,
                'purged': len(rows) - len(train) - len(valid) - len(test),
                'label_intervals_verified': all('label_end_time' in r for r in rows)}
    return train, valid, test, metadata
