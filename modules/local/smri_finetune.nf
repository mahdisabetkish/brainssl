process SMRI_FINETUNE {
    tag "${row.tag}"
    label 'process_gpu'

    input:
    tuple val(row), path(init_run, stageAs: 'init/*')
    path raw
    path processed

    output:
    tuple val(row), path("${row.tag}"), emit: run

    script:
    def init = row.init ? "init='\"${init_run}/final.pt\"'" : ''
    """
    brainssl smri finetune --set out_dir='"${row.tag}"' raw='"${raw}"' \
        index='"${processed}/msd/index.csv"' splits='"${processed}/msd/splits.json"' \
        bilateral=${row.bilateral} budget='"${row.budget}"' iterations=${row.iterations} \
        seed=${row.seed} workers=${params.smri_workers} ${init} ${params.smri_extra_args} ${params.smri_finetune_args}
    """

    stub:
    """
    mkdir -p ${row.tag} && touch ${row.tag}/best.pt ${row.tag}/config.json ${row.tag}/summary.json
    """
}
