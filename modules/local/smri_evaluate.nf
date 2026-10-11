/*
 * Test-split scores per case (Dice, HD95) for one fine-tuned run, written to a
 * new directory so the training run stays an immutable input, with the
 * probability maps of the showcase scans for the dashboard's viewer.
 */
process SMRI_EVALUATE {
    tag "${row.tag}"
    label 'process_gpu'

    input:
    tuple val(row), path(run)
    path raw
    path processed
    path showcase

    output:
    path "${row.tag}_eval", emit: eval

    script:
    """
    brainssl smri evaluate --run ${run} --out ${row.tag}_eval --raw ${raw} \
        --index ${processed}/msd/index.csv --splits ${processed}/msd/splits.json \
        --save-cases ${showcase}
    """

    stub:
    """
    mkdir -p ${row.tag}_eval && touch ${row.tag}_eval/eval_test.json
    """
}
