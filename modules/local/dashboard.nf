/*
 * The data files behind the dashboard: tables and curves from the finished
 * runs, the showcase scans with both models' predictions, and the deployed
 * connectome model. The page itself is static, so publishing this folder as
 * dashboard/data is all an update takes.
 */
process DASHBOARD {
    label 'process_low'

    input:
    path smri_runs, stageAs: 'smri/*'
    path fmri_runs, stageAs: 'fmri/*'
    path choices
    path showcase
    path exports
    path raw
    path processed_msd, stageAs: 'prepared_msd'
    path processed_abide, stageAs: 'prepared_abide'

    output:
    path 'data', emit: data

    script:
    """
    mkdir -p study processed
    for scheme in kfold loso; do
        if [ -d fmri/study_\${scheme}/\${scheme} ]; then
            ln -s ../fmri/study_\${scheme}/\${scheme} study/\${scheme}
            ln -s ../fmri/study_\${scheme}/settings_\${scheme}.json study/
        fi
    done
    ln -s ../prepared_msd/msd processed/msd
    ln -s ../prepared_abide/abide processed/abide
    brainssl dashboard --smri-runs smri --study study --deep-runs fmri --choices ${choices} \
        --showcase-cases ${showcase} --exports ${exports} --raw ${raw} --processed processed --out data
    """

    stub:
    """
    mkdir -p data && touch data/summary.json
    """
}
