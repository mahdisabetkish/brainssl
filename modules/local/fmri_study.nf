/*
 * Every linear connectome model (tangent, log-Euclidean, correlation, and the
 * noise-aware model) on every atlas and every split of one protocol. Features
 * are computed once per split and shared across models and C values.
 */
process FMRI_STUDY {
    tag "${scheme}"
    label 'process_data'

    input:
    val scheme
    path raw
    path processed

    output:
    path "study_${scheme}", emit: study

    script:
    def splits = scheme == 'kfold' ? "--repeats ${params.fmri_repeats} --folds ${params.fmri_folds}" : ''
    """
    brainssl fmri study --scheme ${scheme} ${splits} --atlases ${params.fmri_atlases} \
        --raw ${raw} --processed ${processed} --out study_${scheme}
    """

    stub:
    """
    mkdir -p study_${scheme}/${scheme}
    """
}
