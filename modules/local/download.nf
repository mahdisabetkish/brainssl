/*
 * Raw data for the requested tracks. storeDir keeps the ~9.6 GB download
 * outside the work directory, so a new run, or a run after `nextflow clean`,
 * finds it there instead of fetching it again.
 */
process DOWNLOAD {
    tag "${tracks}"
    label 'process_data'
    storeDir "${params.outdir}/raw_store"

    input:
    val tracks

    output:
    path 'raw', type: 'dir', emit: raw

    script:
    def abide = tracks.contains('fmri') ? 'brainssl download abide --dest raw' : ''
    def msd   = tracks.contains('smri') ? 'brainssl download msd --dest raw' : ''
    """
    mkdir -p raw
    ${abide}
    ${msd}
    """

    stub:
    """
    mkdir -p raw
    """
}
