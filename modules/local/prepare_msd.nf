/*
 * Per-case index (midline, bounding box, intensity statistics) and the fixed
 * test / validation / label-budget splits.
 */
process PREPARE_MSD {
    label 'process_data'

    input:
    path raw

    output:
    path 'processed', emit: processed

    script:
    """
    brainssl prepare msd --raw ${raw} --out processed --workers ${task.cpus}
    """

    stub:
    """
    mkdir -p processed/msd && touch processed/msd/index.csv processed/msd/splits.json
    """
}
