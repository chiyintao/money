import io
p = 'app/strategy/live_models.py'
s = io.open(p, encoding='utf-8').read()

old = "        path = self.root / 'research_v3' / 'training_dataset.jsonl'" + chr(10) +
      "        if not path.is_file():" + chr(10) +
      "            return {'status': 'unavailable', 'reason': 'training_dataset_missing'}"
assert old in s, 'path block'

new_lines = [
    "        # The dataset the models were actually trained from, found by digest.",
    "        #",
    "        # This used to be the fixed name training_dataset.jsonl. Training writes one",
    "        # dataset per tier -- training_dataset_mainstream.jsonl and its speculative",
    "        # sibling -- so once a run used a tier dataset the stored digest could never",
    "        # equal the one in the manifest. verified went false, trained_market returned",
    "        # no symbols at all, the session selected an empty universe, and nothing was",
    "        # ever traded. The failure reported itself as 'the model was not trained on",
    "        # this market' while the models were trained on it exactly.",
    "        expected = {member.manifest.get('dataset_sha256') for member in self.models.values()}",
    "        expected.discard(None)",
    "        root = self.root / 'research_v3'",
    "        candidates = sorted(root.glob('training_dataset*.jsonl')) if root.is_dir() else []",
    "        path = None",
    "        profile = None",
    "        for candidate in candidates:",
    "            try:",
    "                found = cached_profile(candidate, FEATURES)",
    "            except (OSError, ValueError, KeyError):",
    "                continue",
    "            if not expected or found['sha256'] in expected:",
    "                path, profile = candidate, found",
    "                break",
    "        if profile is None:",
    "            return {'status': 'unavailable', 'reason': 'training_dataset_missing'}",
]
new = chr(10).join(new_lines)
s = s.replace(old, new, 1)

# 去掉后面重复的 expected 定义
dup = "        expected = {member.manifest.get('dataset_sha256') for member in self.models.values()}" + chr(10) +
      "        # Whether these bounds can refuse anything."
assert dup in s, 'dup expected'
s = s.replace(dup, "        # Whether these bounds can refuse anything.", 1)

io.open(p, 'w', encoding='utf-8').write(s)
print('feature space now located by digest')