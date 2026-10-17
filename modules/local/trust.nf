/*
 * Interpretability and trust audits: calibration, subgroup and shortcut checks
 * of the connectome classifier, its weight maps and their stability, and for
 * the segmentation network uncertainty, failure analysis, robustness to
 * perturbed images and probes of the bilateral blocks. Trained models are only
 * read. The segmentation part compares the proposed model with the U-Net
 * trained from scratch at the same label budget, for each budget in
 * params.trust_budgets that has both runs.
 */
process TRUST {
    label 'process_gpu'

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
    path 'trust', emit: trust

    script:
    """
    mkdir -p study processed trust
    for scheme in kfold loso; do
        if [ -d fmri/study_\${scheme}/\${scheme} ]; then
            ln -s ../fmri/study_\${scheme}/\${scheme} study/\${scheme}
            ln -s ../fmri/study_\${scheme}/settings_\${scheme}.json study/
        fi
    done
    ln -s ../prepared_msd/msd processed/msd
    ln -s ../prepared_abide/abide processed/abide

    brainssl trust fmri --study study --choices ${choices} --raw ${raw} --out trust/trust_fmri.json
    brainssl trust interpret --connectome ${exports}/connectome --choices ${choices} --raw ${raw} \
        --processed processed --bootstraps ${params.trust_bootstraps} \
        --stability-atlases ${params.trust_stability_atlases} --deletion-atlases ${params.trust_deletion_atlases} \
        --out trust/trust_interpret.json

    RUNS=""; PAIRS=""
    for budget in ${params.trust_budgets.tokenize(',').join(' ')}; do
        if [ -d smri/ft_ours_\${budget} ] && [ -d smri/ft_unet_scratch_\${budget} ]; then
            RUNS="\$RUNS ours_\${budget}=smri/ft_ours_\${budget} scratch_\${budget}=smri/ft_unet_scratch_\${budget}"
            PAIRS="\$PAIRS ours_\${budget}:scratch_\${budget}"
        fi
    done
    if [ -d smri/ft_unet_mim_10pct ]; then RUNS="\$RUNS mim_10pct=smri/ft_unet_mim_10pct"; fi
    if [ -n "\$RUNS" ]; then
        brainssl trust smri --runs \$RUNS --pairs \$PAIRS --robust-cases ${params.trust_robust_cases} \
            --layers-for ${showcase} --layers-root trust/layers --raw ${raw} \
            --index processed/msd/index.csv --splits processed/msd/splits.json --out trust/trust_smri.json
    fi
    """

    stub:
    """
    mkdir -p trust && touch trust/trust_fmri.json trust/trust_interpret.json trust/trust_smri.json
    """
}
