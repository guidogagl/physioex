# ProtoSleepNet (pretrained)

Interpretable, prototype-based sleep staging. This example just loads the
pretrained model from the HuggingFace Hub and runs a forward pass.

- **Paper:** "Prototype-based interpretable sleep staging with physiologically
  meaningful sub-stage pattern discovery" (npj Digital Medicine).
- **Full code / reproduction:** https://github.com/guidogagl/protosleepnet
- **Weights:** `4rooms/physioex` — `protosleepnet-st-3ch-mixer` (PST),
  `protosleepnet-seq-3ch-mixer` (PSN).

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("protosleepnet-st-3ch-mixer", verbose=True)
```
