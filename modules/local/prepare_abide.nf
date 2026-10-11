/*
 * Packed time series per atlas, full-scan shrunk correlations, their matrix
 * logarithms, and the temporal-window connectomes the noise-aware models use.
 */
process PREPARE_ABIDE {
    label 'process_data'

    input:
    path raw

    output:
    path 'processed', emit: processed

    script:
    """
    brainssl prepare abide --raw ${raw} --out processed
    """

    stub:
    """
    mkdir -p processed/abide
    """
}
