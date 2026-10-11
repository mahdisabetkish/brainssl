/*
 * One deep single-atlas baseline (assets/fmri_methods.csv) on the same splits
 * as the linear study.
 */
process FMRI_DEEP {
    tag "${row.method}/${row.scheme}"
    label 'process_gpu'

    input:
    val row
    path raw
    path processed

    output:
    path "${row.method}_${row.scheme}", emit: run

    script:
    """
    brainssl fmri cv --method ${row.method} --scheme ${row.scheme} --raw ${raw} --processed ${processed} \
        --out ${row.method}_${row.scheme} --repeats ${params.fmri_repeats} --folds ${params.fmri_folds} \
        ${row.args ?: ''}
    """

    stub:
    """
    mkdir -p ${row.method}_${row.scheme} && touch ${row.method}_${row.scheme}/summary.json
    """
}
