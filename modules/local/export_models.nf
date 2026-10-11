/*
 * ONNX model and metadata for the Rust binary and the browser, the refitted
 * connectome classifier, the held-out demo bundle and the model cards: the
 * folders that go to the Hugging Face Hub.
 */
process EXPORT_MODELS {
    label 'process_data'

    input:
    path smri_runs, stageAs: 'smri/*'
    path raw
    path processed
    path choices
    path report

    output:
    path 'exports', emit: exports

    script:
    """
    brainssl export all --out exports --smri-run smri/ft_ours_100pct --raw ${raw} \
        --processed ${processed} --choices ${choices} --report ${report}/results.json --folds ${params.fmri_folds}
    """

    stub:
    """
    mkdir -p exports
    """
}
