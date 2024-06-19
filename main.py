from physioex.explain.ari_explainer import ARIExplainer

expl = ARIExplainer(
    model_name="chambon2018",
    dataset_name="dreem",
    ckp_path="models/scl/chambon2018/seqlen=3/dreem/dodo/",
    version="dodo",
    use_cache=True,
    sequence_lenght=3,
    batch_size=32,
)

expl.explain(save_csv=True, plot_pca=True, plot_kmeans=True, n_jobs=5)

expl = ARIExplainer(
    model_name="chambon2018",
    dataset_name="dreem",
    ckp_path="models/scl/chambon2018/seqlen=3/dreem/dodh/",
    version="dodh",
    use_cache=True,
    sequence_lenght=3,
    batch_size=32,
)

expl.explain(save_csv=True, plot_pca=True, plot_kmeans=True, n_jobs=5)

expl = ARIExplainer(
    model_name="seqsleepnet",
    dataset_name="dreem",
    ckp_path="models/scl/seqsleepnet/seqlen=3/dreem/dodo/",
    version="dodo",
    use_cache=True,
    sequence_lenght=3,
    batch_size=32,
)

expl.explain(save_csv=True, plot_pca=True, plot_kmeans=True, n_jobs=5)

expl = ARIExplainer(
    model_name="seqsleepnet",
    dataset_name="dreem",
    ckp_path="models/scl/seqsleepnet/seqlen=3/dreem/dodh/",
    version="dodh",
    use_cache=True,
    sequence_lenght=3,
    batch_size=32,
)

expl.explain(save_csv=True, plot_pca=True, plot_kmeans=True, n_jobs=5)

expl = ARIExplainer(
    model_name="tinysleepnet",
    dataset_name="dreem",
    ckp_path="models/scl/tinysleepnet/seqlen=3/dreem/dodo/",
    version="dodo",
    use_cache=True,
    sequence_lenght=3,
    batch_size=32,
)

expl.explain(save_csv=True, plot_pca=True, plot_kmeans=True, n_jobs=5)

expl = ARIExplainer(
    model_name="tinysleepnet",
    dataset_name="dreem",
    ckp_path="models/scl/tinysleepnet/seqlen=3/dreem/dodh/",
    version="dodh",
    use_cache=True,
    sequence_lenght=3,
    batch_size=32,
)

expl.explain(save_csv=True, plot_pca=True, plot_kmeans=True, n_jobs=5)
