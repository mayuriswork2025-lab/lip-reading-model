"""Character encoding for CTC sentence lip reading (a-z + space + blank)."""

BLANK = 27
NUM_CLASSES = 28


def text_to_labels(text: str) -> list[int]:
    labels = []
    for char in text.lower():
        if "a" <= char <= "z":
            labels.append(ord(char) - ord("a"))
        elif char == " ":
            labels.append(26)
    return labels


def labels_to_text(labels: list[int]) -> str:
    text = ""
    for label in labels:
        if 0 <= label < 26:
            text += chr(label + ord("a"))
        elif label == 26:
            text += " "
    return text


def ctc_greedy_decode(log_probs) -> str:
    """Greedy CTC decode from log-probabilities [T, C]."""
    import torch

    if isinstance(log_probs, torch.Tensor):
        probs = log_probs.detach().cpu()
    else:
        probs = log_probs

    best = probs.argmax(dim=-1).tolist()
    collapsed = []
    previous = None
    for index in best:
        if index != BLANK and index != previous:
            collapsed.append(index)
        previous = index
    return labels_to_text(collapsed)
