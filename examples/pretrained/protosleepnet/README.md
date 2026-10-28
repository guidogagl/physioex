# ProtoSleepNet and ProtoSleepTransformer (pretrained)

Interpretable, prototype-based sleep staging. This example loads a pretrained model from the Hugging Face Hub and runs a
forward pass.

- **Paper:** "Prototype-based interpretable sleep staging with physiologically meaningful sub-stage pattern discovery"
  (npj Digital Medicine).
- **Paper code / analyses:** https://github.com/guidogagl/protosleepnet
- **Weights:** `4rooms/physioex`, folders `protosleepnet-gagliardi` (PSN, SeqSleepNet epoch encoder, L=20) and
  `protosleeptransformer-gagliardi` (PST, SleepTransformer epoch encoder, L=21). Both were trained jointly on 28 sites; the
  documentation page [ProtoSleepNet and ProtoSleepTransformer](https://guidogagl.github.io/physioex/) lists the data, the
  recipe and the per-site test performance. The one-dataset weights of the paper are a separate archive.

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("protosleeptransformer-gagliardi", verbose=True)
```

`export_release.py` builds the Hub folder (`model.pt`, `config.json`, `metrics.json`) from a training checkpoint and its
per-site evaluation JSON. It copies the metrics unchanged and never uploads anything.
