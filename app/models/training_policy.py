import time


def validate_split_sizes(rows, train_ratio, valid_ratio, minimum=5):
    total = len(rows)
    if total < minimum * 3:
        return {'ready': False, 'reason': 'insufficient_split_rows', 'rows': total, 'minimum_total': minimum * 3}
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        return {'ready': False, 'reason': 'invalid_split_ratios'}
    train = int(total * train_ratio)
    validation = int(total * valid_ratio)
    test = total - train - validation
    ready = min(train, validation, test) >= minimum
    return {'ready': ready, 'reason': None if ready else 'split_below_minimum', 'train': train, 'validation': validation, 'test': test, 'minimum': minimum}


def retrain_required(model_manifest, *, now_ms=None, max_age_ms=6 * 60 * 60 * 1000,
                     current_dataset_hash=None, current_feature_version=None, drifted=None):
    """Whether the live model still describes the data arriving now.

    The three original inputs are all facts about the artifact: how old it is, which
    dataset it was fitted on, which feature contract it was built against. None of them
    is a fact about the market. A model can be an hour old, on the current dataset, on the
    current contract, and still be answering questions about a distribution that no longer
    exists -- which is precisely what the drift monitor measures hourly and what nothing
    acted on. ``drifted`` is the list of columns whose live distribution has moved; an
    empty list means checked and clean, and None means not checked at all, which is not
    the same answer.
    """
    if drifted:
        # Ahead of the artifact checks: it is the only one of the four that is about the
        # present rather than about the file.
        return True, 'feature_drift:' + ','.join(str(name) for name in drifted)
    if not model_manifest:
        return True, 'no_model'
    now = int(now_ms or time.time() * 1000)
    created = int(model_manifest.get('created_at', 0))
    if created <= 0 or now - created >= max_age_ms:
        return True, 'model_expired'
    dataset = model_manifest.get('dataset') or {}
    if current_dataset_hash and dataset.get('sha256') != current_dataset_hash:
        return True, 'dataset_changed'
    if current_feature_version and model_manifest.get('feature_version') != current_feature_version:
        return True, 'features_changed'
    return False, 'fresh'
