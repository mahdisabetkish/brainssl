process MAKE_TEST_DATA {
    tag 'synthetic'
    label 'process_low'

    output:
    path 'raw', type: 'dir', emit: raw

    script:
    """
    make_test_data.py --out raw
    """

    stub:
    """
    mkdir -p raw/Task01_BrainTumour raw/abide
    """
}
