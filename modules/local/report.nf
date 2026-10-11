/*
 * Results tables and paired tests, rebuilt from the run artefacts alone.
 */
process REPORT {
    label 'process_low'

    input:
    path smri_runs, stageAs: 'smri/*'
    path fmri_runs, stageAs: 'fmri/*'
    path choices

    output:
    path 'report', emit: report

    script:
    """
    mkdir -p study
    for scheme in kfold loso; do
        if [ -d fmri/study_\${scheme}/\${scheme} ]; then
            ln -s ../fmri/study_\${scheme}/\${scheme} study/\${scheme}
            ln -s ../fmri/study_\${scheme}/settings_\${scheme}.json study/
        fi
    done
    brainssl report --smri-runs smri --study study --deep-runs fmri --choices ${choices} --out report
    """

    stub:
    """
    mkdir -p report && touch report/results.md report/results.json
    """
}
