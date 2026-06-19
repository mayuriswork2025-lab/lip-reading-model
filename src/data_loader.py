from pathlib import Path


def parse_alignment(align_path):
    words = []
    with open(align_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] not in {"sil", "sp"}:
                start = int(parts[0]) // 1000
                end = int(parts[1]) // 1000
                words.append((start, end, parts[2]))
    return words


def discover_word_samples(mouth_crops_dir):
    root = Path(mouth_crops_dir)
    samples = []
    for npy_path in sorted(root.glob("*/*.npy")):
        label = npy_path.parent.name
        if label.startswith("_"):
            continue
        samples.append((str(npy_path), label))
    return samples


def train_val_split(samples, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    if len(set(labels)) < 2 or len(samples) < 4:
        return samples, []

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    return train, val
