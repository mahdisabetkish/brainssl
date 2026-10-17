#!/usr/bin/env nextflow
/*
 * brainssl
 *
 * Two label-efficiency studies on public neuroimaging data, from raw download
 * to the published dashboard:
 *
 *   structural MRI   brain tumour segmentation on MSD Task01 (BraTS), with a
 *                    bilateral U-Net pretrained by contralateral masked modelling
 *   resting fMRI     autism vs control classification on ABIDE I, with
 *                    connectome classifiers trained on each scan's own
 *                    temporal resampling distribution
 *
 *   nextflow run . -profile docker,gpu              # the full study on one GPU machine
 *   nextflow run . -profile slurm,apptainer,gpu     # on a cluster
 *   nextflow run . -profile test,docker             # synthetic data, minutes on a CPU
 *   nextflow run . -profile docker,gpu --raw data/raw --tracks smri
 *
 * The run matrices live in assets/smri_runs.csv and assets/fmri_methods.csv.
 * Adding a run is a new row there, not a code change.
 */

include { MAKE_TEST_DATA } from './modules/local/make_test_data'
include { DOWNLOAD       } from './modules/local/download'
include { PREPARE_MSD    } from './modules/local/prepare_msd'
include { PREPARE_ABIDE  } from './modules/local/prepare_abide'
include { SMRI_PRETRAIN  } from './modules/local/smri_pretrain'
include { SMRI_FINETUNE  } from './modules/local/smri_finetune'
include { SMRI_EVALUATE  } from './modules/local/smri_evaluate'
include { FMRI_STUDY     } from './modules/local/fmri_study'
include { FMRI_DEEP      } from './modules/local/fmri_deep'
include { EXPORT_MODELS  } from './modules/local/export_models'
include { REPORT         } from './modules/local/report'
include { TRUST          } from './modules/local/trust'
include { DASHBOARD      } from './modules/local/dashboard'


def scaledIterations(row) {
    // The test profile shrinks every run to a couple of steps.
    return Math.max(1, ((row.iterations as int) * (params.smri_iter_scale as double)) as int)
}


def readMatrix(path) {
    def rows = file(path, checkIfExists: true).splitCsv(header: true)
    rows.each { row ->
        if( !row.tag && !row.method ) {
            error "run matrix ${path}: every row needs a tag or method"
        }
    }
    return rows
}


workflow SMRI {
    take:
    raw
    processed
    showcase

    main:
    def matrix = readMatrix(params.smri_runs)

    pre = channel.fromList(matrix.findAll { row -> row.stage == 'pretrain' })
        .map { row -> row + [iterations: scaledIterations(row)] }
    SMRI_PRETRAIN(pre, raw, processed)

    checkpoints = SMRI_PRETRAIN.out.run

    finetune = channel.fromList(matrix.findAll { row -> row.stage == 'finetune' })
        .map { row -> row + [iterations: scaledIterations(row)] }

    // A run initialised from a checkpoint waits only for that pretraining run.
    from_scratch = finetune.filter { row -> !row.init }.map { row -> [row, []] }
    from_checkpoint = finetune.filter { row -> row.init }
        .map { row -> [row.init, row] }
        .combine(checkpoints, by: 0)
        .map { _init, row, run -> [row, run] }

    SMRI_FINETUNE(from_scratch.mix(from_checkpoint), raw, processed)
    SMRI_EVALUATE(SMRI_FINETUNE.out.run, raw, processed, showcase)

    emit:
    runs  = SMRI_FINETUNE.out.run.map { _row, run -> run }.mix(SMRI_PRETRAIN.out.run.map { _tag, run -> run })
    evals = SMRI_EVALUATE.out.eval
}


workflow FMRI {
    take:
    raw
    processed

    main:
    FMRI_STUDY(channel.of('kfold', 'loso'), raw, processed)
    deep = channel.fromList(readMatrix(params.fmri_methods))
    FMRI_DEEP(deep, raw, processed)

    emit:
    study = FMRI_STUDY.out.study
    deep  = FMRI_DEEP.out.run
}


workflow {
    def tracks = params.tracks.tokenize(',')

    if( params.test ) {
        raw = MAKE_TEST_DATA().raw
    }
    else if( params.raw ) {
        raw = channel.fromPath(params.raw, type: 'dir', checkIfExists: true).first()
    }
    else {
        raw = DOWNLOAD(tracks.join(',')).raw
    }

    smri_runs = channel.empty()
    smri_evals = channel.empty()
    fmri_runs = channel.empty()
    showcase = channel.fromPath(params.showcase_cases, checkIfExists: true).first()

    if( 'smri' in tracks ) {
        processed_msd = PREPARE_MSD(raw).processed
        SMRI(raw, processed_msd, showcase)
        smri_runs = SMRI.out.runs
        smri_evals = SMRI.out.evals
    }
    if( 'fmri' in tracks ) {
        processed_abide = PREPARE_ABIDE(raw).processed
        FMRI(raw, processed_abide)
        fmri_runs = FMRI.out.study.mix(FMRI.out.deep)
    }

    all_smri = smri_runs.mix(smri_evals).collect().ifEmpty([])
    all_fmri = fmri_runs.collect().ifEmpty([])
    choices  = channel.fromPath(params.fmri_choices, checkIfExists: true).first()

    REPORT(all_smri, all_fmri, choices)

    if( params.export && 'smri' in tracks && 'fmri' in tracks ) {
        EXPORT_MODELS(all_smri, raw, processed_abide, choices, REPORT.out.report)
        TRUST(all_smri, all_fmri, choices, showcase, EXPORT_MODELS.out.exports, raw, processed_msd, processed_abide)
        DASHBOARD(all_smri, all_fmri, choices, showcase, EXPORT_MODELS.out.exports, TRUST.out.trust, raw,
                  processed_msd, processed_abide)
    }

    workflow.onComplete = {
        def message = workflow.success
            ? "Done. Results, tables and dashboard data are in ${params.outdir}"
            : "The run stopped with errors; see ${params.outdir}/pipeline_info/ and .nextflow.log"
        log.info(message)
    }
}
