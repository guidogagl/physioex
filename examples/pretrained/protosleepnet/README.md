# ProtoSleepNet (pretrained)

Interpretable, prototype-based sleep staging. This example just loads the
pretrained model from the HuggingFace Hub and runs a forward pass.

- **Paper:** "Prototype-based interpretable sleep staging with physiologically
  meaningful sub-stage pattern discovery" (npj Digital Medicine).
- **Full code / reproduction:** https://github.com/guidogagl/protosleepnet
- **Weights:** `4rooms/sleep-prototypes` — `protosleeptransformer-gagliardi` (PST),
  `protosleepnet-gagliardi` (PSN). (Naming follows the physioex `<model>-<author>`
  convention; the Phan originals `{seqsleepnet,sleeptransformer}-phan` are in `4rooms/physioex`.)

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("protosleeptransformer-gagliardi", repo_id="4rooms/sleep-prototypes", verbose=True)
```
