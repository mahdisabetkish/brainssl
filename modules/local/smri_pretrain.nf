process SMRI_PRETRAIN {
    tag "${row.tag}"
    label 'process_gpu'

    input:
    val row
    path raw
    path processed

    output:
    tuple val(row.tag), path("${row.tag}"), emit: run

    script:
    """
    brainssl smri pretrain --set out_dir='"${row.tag}"' raw='"${raw}"' \
        index='"${processed}/msd/index.csv"' splits='"${processed}/msd/splits.json"' \
        bilateral=${row.bilateral} masking='"${row.masking}"' iterations=${row.iterations} \
        seed=${row.seed} workers=${params.smri_workers} ${params.smri_extra_args}
    """

    stub:
    """
    mkdir -p ${row.tag} && touch ${row.tag}/final.pt ${row.tag}/summary.json
    """
}
